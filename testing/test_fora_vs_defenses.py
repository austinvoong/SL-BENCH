"""
testing/test_fora_vs_defenses.py
──────────────────────────────────
Evaluate FORA against existing defenses to empirically establish that
NoPeekNN and DP cannot stop a semi-honest reconstruction attack.

This is the key empirical argument for the SL-BENCH thesis:
    "No practical defense protects against FORA while preserving utility."

Five conditions are run:
    A) Vanilla SL           — no defense, FORA baseline
    B) NoPeekNN (λ=0.5)     — statistical independence regularization
    C) DP Gaussian σ/C=0.01 — weak noise (utility-preserving)
    D) DP Gaussian σ/C=0.1  — moderate noise
    E) DP Laplace ε=1.0     — pure DP alternative

For each condition, the FULL FORA three-phase pipeline runs:
    Phase 1: Substitute client trains IN PARALLEL with the defended SL model.
             The substitute sees exactly what the semi-honest server sees —
             meaning it trains on the DEFENDED smashed data (noisy or shaped).
             This is the correct threat model.
    Phase 2: Inverse network trained on auxiliary data after SL completes.
    Phase 3: Snapshot-based reconstruction of private training images.

Key design decision on DP:
    FORA's substitute trains on the NOISY smashed data (what the server
    actually receives). This is the correct and conservative evaluation:
    we are not giving FORA the clean signal. If FORA still reconstructs
    well despite the noise, the defense has genuinely failed.

Usage:
    python testing/test_fora_vs_defenses.py
    python testing/test_fora_vs_defenses.py --conditions vanilla nopeeknn dp_gaussian_weak
    python testing/test_fora_vs_defenses.py --sl-epochs 10 --inverse-epochs 15 --quick
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import get_dataloader
from models import create_split_simple_cnn
from attacks.fora import FORAAttack
from metrics.reconstruction import compute_ssim, compute_psnr, distance_correlation
from defenses.differential_privacy import clip_per_sample, gaussian_noise, laplace_noise
from metrics.reconstruction import distance_correlation


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SMASHED_SHAPE = {
    1: (32, 16, 16),
    2: (64,  8,  8),
    3: (128, 4,  4),
}

# Condition registry: (display_label, defense_type, defense_kwargs)
CONDITIONS = {
    "vanilla": (
        "Vanilla SL (No Defense)",
        "none",
        {},
    ),
    "nopeeknn": (
        "NoPeekNN (λ=0.5)",
        "nopeeknn",
        {"lambda_dcor": 0.5, "dcor_subsample": 64},
    ),
    "dp_gaussian_weak": (
        "DP Gaussian σ/C=0.01",
        "dp_gaussian",
        {"noise_multiplier": 0.01, "clip_norm": 1.0},
    ),
    "dp_gaussian_mid": (
        "DP Gaussian σ/C=0.1",
        "dp_gaussian",
        {"noise_multiplier": 0.10, "clip_norm": 1.0},
    ),
    "dp_laplace": (
        "DP Laplace ε=1.0",
        "dp_laplace",
        {"epsilon": 1.0, "clip_norm": 1.0},
    ),
}

ALL_CONDITION_KEYS = list(CONDITIONS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def make_aux_loader(
    test_dataset,
    batch_size: int,
    aux_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split the test set into auxiliary data (server's public data for FORA)
    and a held-out eval set (for reconstruction quality measurement).

    The auxiliary and eval sets are both from the test set, ensuring they
    are NOT in the private training data — matching the FORA paper's setup.
    """
    n = len(test_dataset)
    n_aux = max(int(n * aux_frac), 256)
    n_eval = n - n_aux
    aux_subset, eval_subset = random_split(
        test_dataset, [n_aux, n_eval],
        generator=torch.Generator().manual_seed(seed),
    )
    aux_loader = DataLoader(
        aux_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return aux_loader, eval_loader


# ─────────────────────────────────────────────────────────────────────────────
# Defense-specific smashed data transformation
# Applied CLIENT-SIDE before the server receives the tensor.
# ─────────────────────────────────────────────────────────────────────────────

def apply_defense_to_smashed(
    smashed: torch.Tensor,
    defense_type: str,
    defense_kwargs: dict,
    images: torch.Tensor,           # only needed for NoPeekNN dCor
    client_optimizer: torch.optim.Optimizer,
    lambda_dcor: float = 0.0,       # NoPeekNN only
    dcor_subsample: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply the defense transformation to smashed data.

    For NoPeekNN:
        Computes and backward()s the dCor loss through smashed data
        (with retain_graph=True) before returning the clean tensor.
        The dCor gradients accumulate in smashed.grad; the server gradient
        will be added on top in the main backward pass.

    For DP:
        Clips per-sample norms and adds calibrated noise. The noisy tensor
        is returned; this is what the server (and FORA) actually sees.

    For no defense:
        Returns smashed unchanged.

    Returns:
        (smashed_to_send, dcor_value_or_None)
        smashed_to_send: the tensor the server receives (may be noisy)
        dcor_value:      raw dCor scalar for logging (NoPeekNN only)
    """
    if defense_type == "nopeeknn":
        # ── NoPeekNN: dCor regularization ────────────────────────────────────
        n = smashed.shape[0]
        if n > dcor_subsample:
            idx = torch.randperm(n, device=smashed.device)[:dcor_subsample]
            smashed_sub = smashed[idx]
            images_sub  = images[idx]
        else:
            smashed_sub = smashed
            images_sub  = images

        dcor_val  = distance_correlation(images_sub.detach(), smashed_sub)
        dcor_loss = lambda_dcor * dcor_val
        # Backward dCor FIRST with retain_graph so the graph stays alive
        # for the subsequent server gradient backward (mirrors nopeeknn.py).
        dcor_loss.backward(retain_graph=True)

        return smashed, dcor_val.detach()

    elif defense_type == "dp_gaussian":
        # ── DP Gaussian: clip + Gaussian noise ───────────────────────────────
        noise_multiplier = defense_kwargs["noise_multiplier"]
        clip_norm        = defense_kwargs["clip_norm"]
        clipped   = clip_per_sample(smashed, clip_norm)
        perturbed = gaussian_noise(clipped, clip_norm, noise_multiplier)
        return perturbed, None

    elif defense_type == "dp_laplace":
        # ── DP Laplace: clip + Laplace noise ─────────────────────────────────
        epsilon  = defense_kwargs["epsilon"]
        clip_norm = defense_kwargs["clip_norm"]
        clipped   = clip_per_sample(smashed, clip_norm)
        perturbed = laplace_noise(clipped, clip_norm, epsilon)
        return perturbed, None

    else:
        # No defense
        return smashed, None


# ─────────────────────────────────────────────────────────────────────────────
# Core: integrated SL + FORA Phase 1 training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_sl_with_fora(
    client_model: nn.Module,
    server_model: nn.Module,
    train_loader: DataLoader,
    fora: FORAAttack,
    epochs: int,
    defense_type: str,
    defense_kwargs: dict,
    device: torch.device,
    lr: float = 1e-3,
) -> dict:
    """
    Vanilla SL training loop with:
        - The specified defense applied at the cut layer
        - FORA Phase 1 (substitute training) running in parallel

    FORA's substitute sees exactly what the server sees — the defended
    smashed data. For DP conditions this means FORA trains on noisy tensors.
    This is the correct (conservative) threat model evaluation.

    Args:
        client_model:   Victim client-side model
        server_model:   Server-side model
        train_loader:   Private training data
        fora:           Initialized FORAAttack (Phase 1 will be called here)
        epochs:         Number of SL training epochs
        defense_type:   One of 'none', 'nopeeknn', 'dp_gaussian', 'dp_laplace'
        defense_kwargs: Defense hyperparameters
        device:         Compute device
        lr:             Learning rate

    Returns:
        Training history dict
    """
    criterion    = nn.CrossEntropyLoss()
    client_opt   = torch.optim.Adam(client_model.parameters(), lr=lr)
    server_opt   = torch.optim.Adam(server_model.parameters(), lr=lr)

    lambda_dcor    = defense_kwargs.get("lambda_dcor", 0.5)
    dcor_subsample = defense_kwargs.get("dcor_subsample", 64)

    history = {
        "train_loss": [], "train_acc": [],
        "fora_sub_loss": [], "fora_disc_loss": [], "fora_mmd_loss": [],
        "dcor_values": [],  # NoPeekNN only; NaN otherwise
    }

    last_epoch = epochs - 1

    for epoch in range(epochs):
        client_model.train()
        server_model.train()

        total_loss  = 0.0
        total_correct = 0
        total_samples = 0
        fora_sub_acc  = []
        fora_disc_acc = []
        fora_mmd_acc  = []
        dcor_acc      = []

        is_last = (epoch == last_epoch)

        pbar = tqdm(
            train_loader,
            desc=f"  [{defense_type}] Epoch {epoch+1:3d}/{epochs}",
            leave=False,
        )

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # ── Client forward ───────────────────────────────────────────────
            client_opt.zero_grad()
            smashed = client_model(images)

            # ── Defense transformation ───────────────────────────────────────
            # Returns (smashed_to_send, optional_dcor_val)
            # For NoPeekNN: smashed_to_send == smashed (clean), dCor already
            # backwarded with retain_graph=True.
            # For DP: smashed_to_send is noisy (what server actually receives).
            smashed_defended, dcor_val = apply_defense_to_smashed(
                smashed=smashed,
                defense_type=defense_type,
                defense_kwargs=defense_kwargs,
                images=images,
                client_optimizer=client_opt,
                lambda_dcor=lambda_dcor,
                dcor_subsample=dcor_subsample,
            )

            if dcor_val is not None:
                dcor_acc.append(dcor_val.item())

            # ── FORA Phase 1: update substitute with defended smashed data ───
            # FORA receives what the server actually sees (noisy if DP).
            fora_metrics = fora.update_substitute(smashed_defended.detach())
            fora_sub_acc.append(fora_metrics["sub_loss"])
            fora_disc_acc.append(fora_metrics["disc_loss"])
            fora_mmd_acc.append(fora_metrics["mmd_loss"])

            # ── Snapshot collection (final epoch only) ───────────────────────
            if is_last:
                fora.add_to_snapshot(smashed_defended.detach())

            # ── Simulate network boundary ────────────────────────────────────
            smashed_server = smashed_defended.detach().requires_grad_(True)

            # ── Server forward + backward ────────────────────────────────────
            server_opt.zero_grad()
            outputs   = server_model(smashed_server)
            task_loss = criterion(outputs, labels)
            task_loss.backward()

            grad_to_client = smashed_server.grad.clone()
            server_opt.step()

            # ── Client backward ──────────────────────────────────────────────
            # For NoPeekNN: dCor grads already in smashed.grad; server grad
            # is added on top here (same pattern as nopeeknn.py).
            # For DP: smashed_defended is a separate tensor (noisy clone),
            # so we backward through smashed_defended → smashed_server.grad
            # arrives at smashed_defended. We need the gradient w.r.t. the
            # original smashed for the client update.
            #
            # For DP the gradient chain: smashed_defended = f(smashed)
            # where f = clip + noise. Noise has zero grad; clip is a scaling
            # operation. So grad flows back through the clip scaling factor.
            # This is the standard "straight-through" for DP-SGD.
            smashed_defended.backward(grad_to_client) if defense_type.startswith("dp") \
                else smashed.backward(grad_to_client)
            client_opt.step()

            # ── Metrics ──────────────────────────────────────────────────────
            _, pred = outputs.max(1)
            bs = labels.size(0)
            total_loss    += task_loss.item() * bs
            total_correct += pred.eq(labels).sum().item()
            total_samples += bs

            pbar.set_postfix({
                "loss": f"{task_loss.item():.4f}",
                "sub":  f"{fora_metrics['sub_loss']:.4f}",
            })

        # ── Epoch summary ─────────────────────────────────────────────────────
        avg_loss = total_loss / total_samples
        avg_acc  = total_correct / total_samples
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)
        history["fora_sub_loss"].append(sum(fora_sub_acc)  / max(len(fora_sub_acc),  1))
        history["fora_disc_loss"].append(sum(fora_disc_acc) / max(len(fora_disc_acc), 1))
        history["fora_mmd_loss"].append(sum(fora_mmd_acc)  / max(len(fora_mmd_acc),  1))
        history["dcor_values"].append(
            sum(dcor_acc) / len(dcor_acc) if dcor_acc else float("nan")
        )

        print(
            f"  [{defense_type}] Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Acc: {avg_acc*100:5.2f}% | "
            f"SubLoss: {history['fora_sub_loss'][-1]:.4f} | "
            f"MMD: {history['fora_mmd_loss'][-1]:.4f}"
            + (f" | dCor: {history['dcor_values'][-1]:.4f}"
               if not float("nan") == history["dcor_values"][-1] else "")
        )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# SL accuracy evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sl(
    client_model: nn.Module,
    server_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return test accuracy for the trained SL model."""
    client_model.eval()
    server_model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = server_model(client_model(images))
        correct += out.max(1)[1].eq(labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_reconstruction(
    client_model: nn.Module,
    fora: FORAAttack,
    eval_loader: DataLoader,
    device: torch.device,
    defense_type: str,
    defense_kwargs: dict,
    n_batches: int = 20,
) -> dict:
    """
    Evaluate FORA reconstruction quality on the eval loader.

    For DP conditions, the reconstruction pipeline must use the NOISY smashed
    data — because that is what the server (and therefore FORA's inverse
    network) operates on. Using clean smashed data at eval time would
    overstate the defense's failure; using noisy data is the correct and
    conservative measurement.
    """
    client_model.eval()
    fora.inverse_net.eval()

    all_ssim  = []
    all_psnr  = []
    all_dcor  = []

    for i, (images, _) in enumerate(eval_loader):
        if i >= n_batches:
            break
        images  = images.to(device)
        smashed = client_model(images)

        # Apply the same defense transformation as during training
        # so FORA operates on the same signal it was trained against.
        if defense_type == "dp_gaussian":
            smashed_in = gaussian_noise(
                clip_per_sample(smashed, defense_kwargs["clip_norm"]),
                defense_kwargs["clip_norm"],
                defense_kwargs["noise_multiplier"],
            )
        elif defense_type == "dp_laplace":
            smashed_in = laplace_noise(
                clip_per_sample(smashed, defense_kwargs["clip_norm"]),
                defense_kwargs["clip_norm"],
                defense_kwargs["epsilon"],
            )
        else:
            # NoPeekNN and vanilla: FORA sees the clean smashed data
            smashed_in = smashed

        recon = fora.inverse_net(smashed_in)
        recon = recon.clamp(-1, 1)

        all_ssim.append(compute_ssim(recon, images).item())
        all_psnr.append(compute_psnr(recon, images).item())

        # dCor: measure leakage in the signal FORA actually received
        with torch.no_grad():
            dcor_val = distance_correlation(
                images[:min(64, images.size(0))],
                smashed_in[:min(64, smashed_in.size(0))],
            ).item()
        all_dcor.append(dcor_val)

    return {
        "ssim": sum(all_ssim) / len(all_ssim),
        "psnr": sum(all_psnr) / len(all_psnr),
        "dcor": sum(all_dcor) / len(all_dcor),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(
    key: str,
    args,
    train_loader: DataLoader,
    aux_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Full pipeline for one defense condition:
        1. Init fresh SL models + FORA attacker
        2. Train SL with defense + FORA Phase 1 in parallel
        3. Evaluate SL accuracy
        4. FORA Phase 2: train inverse network
        5. FORA Phase 3: evaluate reconstruction from snapshot + eval loader

    Returns a dict of metrics for the summary table.
    """
    display_label, defense_type, defense_kwargs = CONDITIONS[key]
    bar = "═" * 65
    print(f"\n{bar}")
    print(f"  CONDITION: {display_label}")
    print(f"{bar}\n")

    # ── Fresh models ──────────────────────────────────────────────────────────
    client_model, server_model = create_split_simple_cnn(
        cut_layer=args.cut_layer, num_classes=10
    )
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    # ── FORA attacker ─────────────────────────────────────────────────────────
    smashed_ch, smashed_sp, _ = SMASHED_SHAPE[args.cut_layer]
    fora = FORAAttack(
        smashed_channels=smashed_ch,
        smashed_spatial=smashed_sp,
        aux_loader=aux_loader,
        cut_layer=args.cut_layer,
        lambda_mmd=args.lambda_mmd,
        device=str(device),
    )

    # ── Phase 1: SL training + substitute training ────────────────────────────
    t0 = time.time()
    sl_history = train_sl_with_fora(
        client_model=client_model,
        server_model=server_model,
        train_loader=train_loader,
        fora=fora,
        epochs=args.sl_epochs,
        defense_type=defense_type,
        defense_kwargs=defense_kwargs,
        device=device,
        lr=args.lr,
    )
    t_sl = time.time() - t0

    # ── SL accuracy ───────────────────────────────────────────────────────────
    test_acc = evaluate_sl(client_model, server_model, eval_loader, device)
    print(f"\n  SL training complete ({t_sl:.0f}s) | Test accuracy: {test_acc*100:.2f}%")
    print(f"  Snapshot buffer: {sum(s.shape[0] for s in fora._snapshot_list):,} samples")

    # Substitute convergence diagnostics
    if sl_history["fora_sub_loss"]:
        sub_init  = sl_history["fora_sub_loss"][0]
        sub_final = sl_history["fora_sub_loss"][-1]
        mmd_init  = sl_history["fora_mmd_loss"][0]
        mmd_final = sl_history["fora_mmd_loss"][-1]
        print(f"  Substitute loss: {sub_init:.4f} → {sub_final:.4f}")
        print(f"  MMD loss:        {mmd_init:.4f} → {mmd_final:.4f} "
              f"({'↓ converging' if mmd_final < mmd_init else '↑ check lr'})")

    # ── Phase 2: Train inverse network ────────────────────────────────────────
    print(f"\n  FORA Phase 2 — training inverse network ({args.inverse_epochs} epochs)")
    t_inv = time.time()
    fora.train_inverse_network(epochs=args.inverse_epochs, verbose=False)
    print(f"  Inverse network training: {time.time() - t_inv:.0f}s")

    # ── Phase 3: Evaluate reconstruction ─────────────────────────────────────
    print(f"\n  FORA Phase 3 — evaluating reconstruction quality")
    recon_metrics = eval_reconstruction(
        client_model=client_model,
        fora=fora,
        eval_loader=eval_loader,
        device=device,
        defense_type=defense_type,
        defense_kwargs=defense_kwargs,
        n_batches=args.eval_batches,
    )

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  {display_label:<51} │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Test Accuracy  : {test_acc*100:6.2f}%                          │")
    print(f"  │  SSIM           : {recon_metrics['ssim']:.4f}  (1.0 = perfect recon)     │")
    print(f"  │  PSNR           : {recon_metrics['psnr']:5.2f} dB                        │")
    print(f"  │  dCor (leakage) : {recon_metrics['dcor']:.4f}  (1.0 = full leakage)     │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # ── Save FORA checkpoint ──────────────────────────────────────────────────
    safe_key = key.replace("_", "-")
    fora.save(os.path.join(args.save_dir, f"fora_{safe_key}.pt"))
    torch.save(client_model.state_dict(),
               os.path.join(args.save_dir, f"client_{safe_key}.pt"))

    return {
        "test_accuracy":   test_acc,
        "ssim":            recon_metrics["ssim"],
        "psnr":            recon_metrics["psnr"],
        "dcor":            recon_metrics["dcor"],
        "final_sub_loss":  sl_history["fora_sub_loss"][-1] if sl_history["fora_sub_loss"] else float("nan"),
        "final_mmd_loss":  sl_history["fora_mmd_loss"][-1]  if sl_history["fora_mmd_loss"]  else float("nan"),
        "final_dcor_train": sl_history["dcor_values"][-1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    bar = "═" * 65
    print(f"\n\n{bar}")
    print(f"  SUMMARY  —  FORA Attack vs. Defenses  (Cut Layer {list(results.values())[0].get('cut_layer', '?')})")
    print(f"{bar}\n")

    col_w = 24
    fmt   = f"  {{:<{col_w}}} {{:>9}} {{:>8}} {{:>11}} {{:>9}}"
    div   = "  " + "─" * (col_w + 41)

    print(fmt.format("Defense", "Test Acc", "SSIM ↑", "PSNR (dB) ↑", "dCor ↓"))
    print(div)

    for key, res in results.items():
        label = CONDITIONS[key][0]
        print(fmt.format(
            label[:col_w],
            f"{res['test_accuracy']*100:.2f}%",
            f"{res['ssim']:.4f}",
            f"{res['psnr']:.2f}",
            f"{res['dcor']:.4f}",
        ))

    print(div)

    # ── Relative comparison vs vanilla ──────────────────────────────────────
    if "vanilla" not in results:
        return

    va = results["vanilla"]
    print(f"\n  Effect relative to Vanilla SL:")
    print(f"\n  {'Defense':<{col_w}} {'Acc Δ':>8}  {'SSIM Δ':>9}  {'PSNR Δ':>9}  "
          f"{'dCor Δ':>9}  {'<5pp?':>6}  {'Stops FORA?':>12}")
    print("  " + "─" * (col_w + 66))

    for key, res in results.items():
        if key == "vanilla":
            continue
        label     = CONDITIONS[key][0]
        acc_delta = (res["test_accuracy"] - va["test_accuracy"]) * 100
        ssim_delta = va["ssim"] - res["ssim"]   # positive = defense reduces recon quality
        psnr_delta = va["psnr"] - res["psnr"]
        dcor_delta = va["dcor"] - res["dcor"]
        within_5pp = "✓" if abs(acc_delta) < 5 else "✗"

        # FORA is "stopped" if SSIM drops below 0.2 (below recognizable threshold)
        # and the attack SSIM drop is meaningful (>0.1 reduction from vanilla)
        fora_stopped = "✓ Stopped" if res["ssim"] < 0.2 and ssim_delta > 0.1 else "✗ Bypassed"

        print(
            f"  {label[:col_w]:<{col_w}} "
            f"{acc_delta:>+7.2f}pp  "
            f"{ssim_delta:>+9.4f}  "
            f"{psnr_delta:>+9.2f}  "
            f"{dcor_delta:>+9.4f}  "
            f"{within_5pp:>6}  "
            f"{fora_stopped:>12}"
        )

    print(f"""
  Interpretation guide:
    Acc Δ < -5pp  → defense costs too much accuracy (fails utility target)
    SSIM Δ > 0.10 → reconstruction quality meaningfully degraded (partial win)
    SSIM Δ > 0.30 → reconstruction strongly impaired (significant protection)
    dCor Δ > 0    → less statistical leakage (defense is shaping representations)

    "Stops FORA" requires SSIM < 0.2 AND reduction > 0.1 from baseline.
    Anything less is a bypass — the attacker can still reconstruct usable images.

  Thesis implication:
    If NoPeekNN and DP are shown as "✗ Bypassed" while costing accuracy,
    this motivates AFO as a targeted defense that directly disrupts FORA's
    feature-space learning rather than applying generic noise or correlation
    penalties.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'═'*65}")
    print(f"  FORA Attack vs. Defenses — SL-BENCH Evaluation")
    print(f"{'═'*65}")
    print(f"  Device        : {device}")
    print(f"  Cut layer     : {args.cut_layer}  "
          f"(smashed shape: {SMASHED_SHAPE[args.cut_layer]})")
    print(f"  SL epochs     : {args.sl_epochs}")
    print(f"  Inverse epochs: {args.inverse_epochs}")
    print(f"  λ_mmd         : {args.lambda_mmd}")
    print(f"  Conditions    : {args.conditions}")
    print(f"  Save dir      : {args.save_dir}")
    print(f"{'═'*65}")

    # ── Data ──────────────────────────────────────────────────────────────────
    import torchvision
    import torchvision.transforms as transforms

    _transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset_raw = torchvision.datasets.CIFAR10(
        root=os.path.join(ROOT_DIR, "data"), train=False,
        download=True, transform=_transform_test,
    )

    train_loader, _ = get_dataloader("cifar10", batch_size=args.batch_size)
    aux_loader, eval_loader = make_aux_loader(
        test_dataset_raw, batch_size=args.batch_size, aux_frac=args.aux_frac
    )

    print(f"\n  Data split:")
    print(f"    Private (train)   : {len(train_loader.dataset):,} images")
    print(f"    Auxiliary (server): {len(aux_loader.dataset):,} images")
    print(f"    Evaluation        : {len(eval_loader.dataset):,} images")

    # ── Run conditions ────────────────────────────────────────────────────────
    results = {}
    for key in args.conditions:
        if key not in CONDITIONS:
            print(f"  [WARNING] Unknown condition '{key}', skipping.")
            continue
        res = run_condition(
            key=key,
            args=args,
            train_loader=train_loader,
            aux_loader=aux_loader,
            eval_loader=eval_loader,
            device=device,
        )
        res["cut_layer"] = args.cut_layer
        results[key] = res

    # ── Summary + save ────────────────────────────────────────────────────────
    print_summary(results)

    json_path = os.path.join(args.save_dir, "fora_vs_defenses_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SL-BENCH: FORA attack evaluated against NoPeekNN and DP defenses"
    )

    parser.add_argument(
        "--conditions", nargs="+",
        choices=ALL_CONDITION_KEYS,
        default=ALL_CONDITION_KEYS,
        help=(
            "Which conditions to run. Default: all five. "
            "E.g. --conditions vanilla nopeeknn dp_gaussian_weak"
        ),
    )

    # SL / architecture
    parser.add_argument("--cut-layer",      type=int,   default=2,
                        help="SimpleCNN cut layer (1–3). Default: 2")
    parser.add_argument("--sl-epochs",      type=int,   default=15,
                        help="SL training epochs per condition. Default: 15")
    parser.add_argument("--lr",             type=float, default=1e-3,
                        help="SL learning rate. Default: 1e-3")
    parser.add_argument("--batch-size",     type=int,   default=128,
                        help="Batch size. Default: 128")

    # FORA
    parser.add_argument("--inverse-epochs", type=int,   default=30,
                        help="Inverse network training epochs. Default: 30")
    parser.add_argument("--lambda-mmd",     type=float, default=1.0,
                        help="MK-MMD weight in FORA substitute loss. Default: 1.0")
    parser.add_argument("--aux-frac",       type=float, default=0.1,
                        help="Fraction of test set used as FORA auxiliary data. Default: 0.1")
    parser.add_argument("--eval-batches",   type=int,   default=20,
                        help="Batches used for reconstruction eval. Default: 20")

    # Convenience
    parser.add_argument("--quick",          action="store_true",
                        help="Smoke test: 2 SL epochs, 5 inverse epochs.")
    parser.add_argument("--save-dir",       type=str,
                        default=os.path.join(ROOT_DIR, "results", "fora_vs_defenses"),
                        help="Output directory. Default: results/fora_vs_defenses/")

    args = parser.parse_args()

    if args.quick:
        print("\n[QUICK MODE] sl-epochs=2, inverse-epochs=5")
        args.sl_epochs      = 2
        args.inverse_epochs = 5

    main(args)