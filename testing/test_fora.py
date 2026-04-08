"""
testing/test_fora.py
─────────────────────
Integration test: FORA attack against Vanilla Split Learning on CIFAR-10.

What this test covers:
    - Full FORA three-phase pipeline integrated with VanillaSL training
    - Phase 1: substitute client trains in parallel with SL
    - Phase 2: inverse network trained after SL completes
    - Phase 3: snapshot-based reconstruction of private training images
    - Evaluation: SSIM, PSNR, substitute cosine similarity
    - Comparison: FORA vs. baseline Inverse Network attack on the same model

No defenses are applied in this test. The goal is to:
    1. Confirm FORA implementation is functionally correct
    2. Establish baseline reconstruction numbers for later defense comparison
    3. Verify FORA outperforms the naive Inverse Network attack

Expected behavior (from the paper, CIFAR-10 layer 2):
    - FORA SSIM ~ 0.83, PSNR ~ 22.19 dB
    - Inverse Network (UnSplit baseline) SSIM ~ 0.10, PSNR ~ 10.5 dB
    Note: Our SimpleCNN is smaller than the paper's MobileNet/ResNet-18,
    so results will differ but the ordering should hold.

Usage:
    cd <project_root>
    python testing/test_fora.py [--cut_layer 2] [--epochs 20] [--quick]
"""

import argparse
import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from project root regardless of working directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

from models.simple_cnn import create_split_simple_cnn
from attacks.fora import FORAAttack
from attacks.inverse_network import InverseNetworkAttack
from metrics.reconstruction import compute_ssim, compute_psnr


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test FORA attack against Vanilla Split Learning"
    )
    parser.add_argument(
        "--cut_layer", type=int, default=2, choices=[1, 2, 3],
        help="Cut layer for split (default: 2). "
             "Layer 2 = (64,8,8) smashed data [CIFAR-10 baseline in paper]"
    )
    parser.add_argument(
        "--sl_epochs", type=int, default=20,
        help="Epochs of split learning training (default: 20). "
             "More epochs = better SL model = harder reconstruction."
    )
    parser.add_argument(
        "--inverse_epochs", type=int, default=30,
        help="Epochs for inverse network training after SL (default: 30)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for SL training (default: 128)"
    )
    parser.add_argument(
        "--aux_frac", type=float, default=0.1,
        help="Fraction of CIFAR-10 TEST set to use as auxiliary data "
             "(the 'public' data the server has access to). "
             "Default: 0.1 = 1000 images from a different split than private data."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke-test: 2 SL epochs, 5 inverse epochs, 500 training samples. "
             "Useful for verifying the pipeline runs end-to-end before full training."
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory for CIFAR-10 download (default: ./data)"
    )
    parser.add_argument(
        "--lambda_mmd", type=float, default=1.0,
        help="Weight for MK-MMD term in FORA substitute training (default: 1.0)"
    )
    parser.add_argument(
        "--compare_baseline", action="store_true", default=True,
        help="Also run baseline Inverse Network attack for comparison (default: True)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10(data_dir: str, batch_size: int):
    """
    Load CIFAR-10 and split into three non-overlapping sets:
        - Private data:   training set (client's private data in SL)
        - Auxiliary data: small slice of test set (server's 'public' data)
        - Eval data:      remaining test set (for reconstruction quality eval)

    The auxiliary and eval splits are both from the test set to ensure they
    are NOT in the private training set, matching the paper's setup.
    """
    # Standard CIFAR-10 normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return train_dataset, test_dataset


def make_loaders(
    train_dataset,
    test_dataset,
    batch_size: int,
    aux_frac: float,
    train_n: int = None,
):
    """
    Create data loaders for private, auxiliary, and evaluation data.

    Args:
        train_dataset: Full training set (private data)
        test_dataset:  Full test set (split into auxiliary + eval)
        batch_size:    Batch size
        aux_frac:      Fraction of test set used as server auxiliary data
        train_n:       If set, subsample the training set to this size

    Returns:
        train_loader, aux_loader, eval_loader
    """
    # Private training data
    if train_n is not None:
        train_dataset = Subset(train_dataset, range(train_n))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    # Split test set: auxiliary (server) / eval (for reconstruction metrics)
    n_test = len(test_dataset)
    n_aux  = max(int(n_test * aux_frac), 256)
    n_eval = n_test - n_aux

    aux_subset, eval_subset = random_split(
        test_dataset, [n_aux, n_eval],
        generator=torch.Generator().manual_seed(42)
    )

    aux_loader = DataLoader(
        aux_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\nData split:")
    print(f"  Private training  : {len(train_dataset):,} images")
    print(f"  Auxiliary (server): {n_aux:,} images  "
          f"(from test set, different distribution possible)")
    print(f"  Evaluation        : {n_eval:,} images")

    return train_loader, aux_loader, eval_loader


# ─────────────────────────────────────────────────────────────────────────────
# Smashed data shape lookup for SimpleCNN
# ─────────────────────────────────────────────────────────────────────────────

SMASHED_SHAPE = {
    1: (32, 16, 16),
    2: (64,  8,  8),
    3: (128, 4,  4),
}


# ─────────────────────────────────────────────────────────────────────────────
# Manual SL training loop with FORA integration
# ─────────────────────────────────────────────────────────────────────────────

def train_sl_with_fora(
    client_model: nn.Module,
    server_model: nn.Module,
    train_loader: DataLoader,
    fora: FORAAttack,
    epochs: int,
    device: torch.device,
) -> dict:
    """
    Vanilla split learning training loop with FORA Phase 1 running in parallel.

    At each training step:
        1. Client forward → smashed_data
        2. FORA substitute update using that smashed data [Phase 1]
        3. Server forward + backward (standard SL)
        4. Client backward (standard SL)

    The FORA update happens BEFORE the SL update because the SL update
    modifies the client model, and FORA should see the current iteration's
    smashed data. (In practice, the order does not significantly matter since
    FORA's substitute and discriminator are separate networks.)

    Returns:
        Training history with SL and FORA metrics
    """
    criterion       = nn.CrossEntropyLoss()
    client_opt      = torch.optim.Adam(client_model.parameters(), lr=1e-3)
    server_opt      = torch.optim.Adam(server_model.parameters(), lr=1e-3)

    history = {
        "sl_train_loss":  [],
        "sl_train_acc":   [],
        "sl_test_acc":    [],
        "fora_sub_loss":  [],
        "fora_disc_loss": [],
        "fora_mmd_loss":  [],
    }

    # Track which epoch is the last (for snapshot collection)
    last_epoch = epochs - 1

    print(f"\n{'=' * 65}")
    print(f"  Vanilla SL Training + FORA Phase 1 (parallel substitute training)")
    print(f"{'=' * 65}")
    print(f"  Cut layer      : {list(SMASHED_SHAPE.keys())} → {fora.cut_layer}")
    print(f"  SL epochs      : {epochs}")
    print(f"  FORA λ_mmd     : {fora.lambda_mmd}")
    print(f"{'=' * 65}\n")

    for epoch in range(epochs):
        client_model.train()
        server_model.train()

        total_sl_loss = 0.0
        total_correct = 0
        total_samples = 0
        fora_sub_losses  = []
        fora_disc_losses = []
        fora_mmd_losses  = []

        is_last_epoch = (epoch == last_epoch)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # ── 1. Client forward ──────────────────────────────────────────────
            client_opt.zero_grad()
            smashed = client_model(images)  # (B, C, H, W)

            # ── 2. FORA Phase 1: update substitute with this smashed batch ─────
            fora_metrics = fora.update_substitute(smashed.detach())
            fora_sub_losses.append(fora_metrics["sub_loss"])
            fora_disc_losses.append(fora_metrics["disc_loss"])
            fora_mmd_losses.append(fora_metrics["mmd_loss"])

            # ── 3. Collect snapshot on FINAL epoch ────────────────────────────
            # Paper: "The server keeps a snapshot Zsnap = Fc(Xpriv) of all
            # smashed data output by the target client under the final training
            # iteration for reconstruction."
            if is_last_epoch:
                fora.add_to_snapshot(smashed.detach())

            # ── 4. Server forward ──────────────────────────────────────────────
            smashed_server = smashed.detach().requires_grad_(True)
            server_opt.zero_grad()
            outputs = server_model(smashed_server)
            loss    = criterion(outputs, labels)
            loss.backward()
            grad_to_client = smashed_server.grad.clone()
            server_opt.step()

            # ── 5. Client backward ─────────────────────────────────────────────
            smashed.backward(grad_to_client)
            client_opt.step()

            # ── Metrics ────────────────────────────────────────────────────────
            _, predicted   = outputs.max(1)
            total_sl_loss += loss.item() * labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        # ── Epoch summary ──────────────────────────────────────────────────────
        avg_sl_loss = total_sl_loss / total_samples
        avg_train_acc = total_correct / total_samples
        avg_sub_loss  = sum(fora_sub_losses) / max(len(fora_sub_losses), 1)
        avg_disc_loss = sum(fora_disc_losses) / max(len(fora_disc_losses), 1)
        avg_mmd_loss  = sum(fora_mmd_losses) / max(len(fora_mmd_losses), 1)

        history["sl_train_loss"].append(avg_sl_loss)
        history["sl_train_acc"].append(avg_train_acc)
        history["fora_sub_loss"].append(avg_sub_loss)
        history["fora_disc_loss"].append(avg_disc_loss)
        history["fora_mmd_loss"].append(avg_mmd_loss)

        print(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"SL Loss: {avg_sl_loss:.4f} | "
            f"Train Acc: {avg_train_acc*100:5.2f}% | "
            f"Sub Loss: {avg_sub_loss:.4f} | "
            f"Disc Loss: {avg_disc_loss:.4f} | "
            f"MMD: {avg_mmd_loss:.4f}"
        )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# SL accuracy evaluation (no FORA involvement)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sl_accuracy(
    client_model: nn.Module,
    server_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate SL classification accuracy on a data loader."""
    criterion = nn.CrossEntropyLoss()
    client_model.eval()
    server_model.eval()

    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        smashed = client_model(images)
        outputs = server_model(smashed)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)

    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_reconstruction(
    client_model: nn.Module,
    reconstruct_fn,
    eval_loader: DataLoader,
    device: torch.device,
    label: str,
    n_batches: int = 20,
) -> dict:
    """
    Evaluate reconstruction quality over n_batches of the eval loader.

    Args:
        client_model:   Victim client model
        reconstruct_fn: Callable(smashed) → reconstructed
        eval_loader:    Evaluation data loader
        device:         Compute device
        label:          Name for display
        n_batches:      Limit number of batches (for speed)

    Returns:
        Dict with ssim and psnr
    """
    client_model.eval()

    all_ssim = []
    all_psnr = []

    for i, (images, _) in enumerate(eval_loader):
        if i >= n_batches:
            break
        images  = images.to(device)
        smashed = client_model(images)
        recon   = reconstruct_fn(smashed)

        # Clamp reconstruction to valid range for SSIM/PSNR
        recon = recon.clamp(-1, 1)

        all_ssim.append(compute_ssim(recon, images).item())
        all_psnr.append(compute_psnr(recon, images).item())

    ssim = sum(all_ssim) / len(all_ssim)
    psnr = sum(all_psnr) / len(all_psnr)

    print(f"\n  {'─' * 50}")
    print(f"  {label}")
    print(f"  {'─' * 50}")
    print(f"    SSIM : {ssim:.4f}   (1.0 = perfect reconstruction)")
    print(f"    PSNR : {psnr:.2f} dB")
    print(f"  {'─' * 50}")

    return {"ssim": ssim, "psnr": psnr}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Inverse Network comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_inverse_network(
    client_model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    cut_layer: int,
    device: torch.device,
    epochs: int = 30,
) -> dict:
    """
    Train and evaluate a basic Inverse Network attack as a baseline.

    This is the naive attacker who simply trains a decoder on smashed data.
    Unlike FORA, this attacker does NOT use domain adaptation to mimic the
    client's representation preferences. It serves as a weaker baseline.

    Expected outcome: lower SSIM/PSNR than FORA, confirming that FORA's
    substitute model provides meaningful improvement.
    """
    print(f"\n{'─' * 65}")
    print(f"  Baseline: Inverse Network Attack")
    print(f"{'─' * 65}")

    baseline = InverseNetworkAttack(
        client_model=client_model,
        cut_layer=cut_layer,
        device=str(device),
    )

    # Build smashed dataset from training data (server's view)
    smashed_dataset = baseline.build_smashed_dataset(
        train_loader, max_samples=5000
    )

    # Train the inverse network
    baseline.train(
        train_dataset=smashed_dataset,
        val_loader=eval_loader,
        epochs=epochs,
        verbose=False,  # suppress per-epoch prints for cleaner output
    )

    # Final evaluation
    ssim, psnr = baseline._evaluate(eval_loader)

    print(f"\n  Baseline Inverse Network final metrics:")
    print(f"    SSIM : {ssim:.4f}")
    print(f"    PSNR : {psnr:.2f} dB")

    return {"ssim": ssim, "psnr": psnr}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick:
        print("\n[QUICK MODE] Overriding: sl_epochs=2, inverse_epochs=5, 500 train samples")
        args.sl_epochs      = 2
        args.inverse_epochs = 5
        train_n             = 500
    else:
        train_n = None

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'=' * 65}")
    print(f"  FORA Attack Test — Vanilla Split Learning on CIFAR-10")
    print(f"{'=' * 65}")
    print(f"  Device      : {device}")
    print(f"  Cut layer   : {args.cut_layer}  "
          f"(smashed shape: {SMASHED_SHAPE[args.cut_layer]})")
    print(f"  SL epochs   : {args.sl_epochs}")
    print(f"  Inverse ep  : {args.inverse_epochs}")
    print(f"  λ_mmd       : {args.lambda_mmd}")
    print(f"{'=' * 65}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset, test_dataset = load_cifar10(args.data_dir, args.batch_size)
    train_loader, aux_loader, eval_loader = make_loaders(
        train_dataset, test_dataset,
        batch_size=args.batch_size,
        aux_frac=args.aux_frac,
        train_n=train_n,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    client_model, server_model = create_split_simple_cnn(
        cut_layer=args.cut_layer, num_classes=10
    )
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    n_client = sum(p.numel() for p in client_model.parameters())
    n_server = sum(p.numel() for p in server_model.parameters())
    print(f"\n  Client model params : {n_client:,}")
    print(f"  Server model params : {n_server:,}")

    # ── FORA attack setup ─────────────────────────────────────────────────────
    smashed_ch, smashed_sp_h, smashed_sp_w = SMASHED_SHAPE[args.cut_layer]
    assert smashed_sp_h == smashed_sp_w, "Expected square smashed data"

    fora = FORAAttack(
        smashed_channels=smashed_ch,
        smashed_spatial=smashed_sp_h,
        aux_loader=aux_loader,
        cut_layer=args.cut_layer,
        lambda_mmd=args.lambda_mmd,
        device=str(device),
    )

    n_sub   = sum(p.numel() for p in fora.substitute.parameters())
    n_disc  = sum(p.numel() for p in fora.discriminator.parameters())
    n_inv   = sum(p.numel() for p in fora.inverse_net.parameters())
    print(f"\n  FORA substitute params    : {n_sub:,}")
    print(f"  FORA discriminator params : {n_disc:,}")
    print(f"  FORA inverse net params   : {n_inv:,}")

    # ── Phase 1: SL training with parallel FORA substitute training ───────────
    t_start = time.time()
    sl_history = train_sl_with_fora(
        client_model=client_model,
        server_model=server_model,
        train_loader=train_loader,
        fora=fora,
        epochs=args.sl_epochs,
        device=device,
    )
    t_sl = time.time() - t_start

    # Final SL accuracy
    test_acc = evaluate_sl_accuracy(client_model, server_model, eval_loader, device)
    print(f"\n  SL Training complete in {t_sl:.1f}s")
    print(f"  Final SL Test Accuracy: {test_acc * 100:.2f}%")

    # Substitute quality (how well it mimics the client)
    print(f"\n  Measuring substitute client quality...")
    fora.measure_substitute_quality(client_model, eval_loader, n_batches=5)

    # ── Phase 2: Train inverse network ────────────────────────────────────────
    t_inv_start = time.time()
    fora.train_inverse_network(epochs=args.inverse_epochs, verbose=True)
    t_inv = time.time() - t_inv_start
    print(f"  Inverse network training: {t_inv:.1f}s")

    # ── Phase 3: Reconstruct from snapshots + evaluate ────────────────────────
    print(f"\n  Snapshot buffer contains "
          f"{sum(s.shape[0] for s in fora._snapshot_list):,} smashed data samples")

    # FORA: reconstruct via f_c^{-1}(Zsnap)
    print(f"\n  Evaluating FORA reconstruction quality...")
    fora_results = eval_reconstruction(
        client_model=client_model,
        reconstruct_fn=lambda z: fora.reconstruct_batch(z),
        eval_loader=eval_loader,
        device=device,
        label="FORA — Feature-Oriented Reconstruction Attack",
        n_batches=20,
    )

    # ── Baseline comparison ───────────────────────────────────────────────────
    baseline_results = None
    if args.compare_baseline:
        print(f"\n  Running baseline Inverse Network attack for comparison...")
        baseline_results = run_baseline_inverse_network(
            client_model=client_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            cut_layer=args.cut_layer,
            device=device,
            epochs=args.inverse_epochs,
        )

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 65}")
    print(f"  FINAL RESULTS — Cut Layer {args.cut_layer} | CIFAR-10")
    print(f"{'=' * 65}")
    print(f"  SL Model Test Accuracy        : {test_acc * 100:.2f}%")
    print(f"")
    print(f"  Attack              │ SSIM ↑  │ PSNR ↑")
    print(f"  ─────────────────── ┼─────────┼─────────")
    print(f"  FORA (ours)         │ {fora_results['ssim']:.4f}  │ {fora_results['psnr']:.2f} dB")
    if baseline_results:
        print(f"  Inverse Network     │ {baseline_results['ssim']:.4f}  │ {baseline_results['psnr']:.2f} dB")

    if baseline_results:
        ratio = fora_results["psnr"] / max(baseline_results["psnr"], 0.01)
        print(f"\n  PSNR improvement (FORA / Baseline): {ratio:.2f}×")
        print(f"  Paper reports ~1.97× (CIFAR-10, layer 2)")

    print(f"\n{'=' * 65}")
    print(f"  Interpretation:")
    print(f"  SSIM > 0.5 : visually recognizable reconstruction → attack succeeds")
    print(f"  SSIM < 0.3 : noisy / unrecognizable → attack weak at this cut layer")
    print(f"{'=' * 65}\n")

    # ── Log FORA training stats ────────────────────────────────────────────────
    if sl_history["fora_sub_loss"]:
        initial_sub  = sl_history["fora_sub_loss"][0]
        final_sub    = sl_history["fora_sub_loss"][-1]
        initial_mmd  = sl_history["fora_mmd_loss"][0]
        final_mmd    = sl_history["fora_mmd_loss"][-1]
        print(f"  Substitute training convergence:")
        print(f"    Sub loss  : {initial_sub:.4f} → {final_sub:.4f}")
        print(f"    MMD loss  : {initial_mmd:.4f} → {final_mmd:.4f} "
              f"({'↓ converging' if final_mmd < initial_mmd else '↑ check lr'})")

    return {
        "fora": fora_results,
        "baseline": baseline_results,
        "sl_accuracy": test_acc,
    }


if __name__ == "__main__":
    main()