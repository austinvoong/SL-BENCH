"""
testing/test_inverse.py

Test the Inverse Network Attack against Vanilla Split Learning.

Demonstrates that a SEMI-HONEST SERVER can reconstruct private training data
post-hoc by training a decoder on smashed data — with zero protocol deviation.

Protocol:
    1. Run vanilla split learning to convergence (honest training)
    2. Server collects (smashed_data, _) pairs throughout training — legitimate
    3. Server trains InverseNetwork: smashed → reconstructed image  (post-hoc)
    4. Evaluate: SSIM, PSNR, distance correlation

This is the BASELINE attack against which FSHA and FORA are compared.
If even this baseline achieves meaningful reconstruction, it establishes that
smashed data leaks significant information even against a semi-honest attacker.

Usage (from project root):
    python testing/test_inverse.py
    python testing/test_inverse.py --sl-epochs 20 --attack-epochs 40 --cut-layer 2
    python testing/test_inverse.py --device cpu --sl-epochs 5 --attack-epochs 10  # quick

Expected results (cut_layer=2, 15 SL epochs, 30 attack epochs):
    Test accuracy  : ~75–80%  (split learning converges normally)
    Attack SSIM    : ~0.20–0.40  (partial reconstruction)
    Attack PSNR    : ~12–18 dB
    Distance corr. : >0.5  (significant leakage in smashed data)
"""

import argparse
import json
import os
import sys

# ── Ensure project root is on path when run directly ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from data import get_dataloader
from models import create_split_simple_cnn
from trainers import VanillaSplitTrainer
from attacks import InverseNetworkAttack
from metrics import reconstruction_report


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SL-BENCH: Test Inverse Network Attack vs Vanilla Split Learning"
    )
    # Split learning
    p.add_argument("--sl-epochs",         type=int,   default=15,
                   help="SL training epochs. Default: 15")
    p.add_argument("--cut-layer",         type=int,   default=2,
                   help="SimpleCNN cut layer (1–3). Default: 2")
    p.add_argument("--lr",                type=float, default=1e-3,
                   help="SL learning rate. Default: 1e-3")
    # Inverse network attack
    p.add_argument("--attack-epochs",     type=int,   default=30,
                   help="Inverse network training epochs. Default: 30")
    p.add_argument("--attack-lr",         type=float, default=1e-3,
                   help="Inverse network learning rate. Default: 1e-3")
    p.add_argument("--attack-batch-size", type=int,   default=128,
                   help="Inverse network training batch size. Default: 128")
    # Data / training
    p.add_argument("--batch-size",        type=int,   default=128,
                   help="SL training batch size. Default: 128")
    # I/O
    p.add_argument("--save-dir",          type=str,   default="results/inverse",
                   help="Directory for outputs. Default: results/inverse")
    p.add_argument("--device",            type=str,   default=None,
                   help="Compute device (cuda/mps/cpu). Default: auto")
    p.add_argument("--skip-sl-training",  action="store_true",
                   help="Skip SL training and load from checkpoint (must exist)")
    return p.parse_args()


def _resolve_device(requested: str) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _print_section(title: str):
    bar = "─" * 65
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = _resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_sl   = os.path.join(args.save_dir, "vanilla_sl.pt")
    ckpt_atk  = os.path.join(args.save_dir, "inverse_net.pt")
    json_path = os.path.join(args.save_dir, "inverse_results.json")

    print("\n" + "═" * 65)
    print("  SL-BENCH  ·  Inverse Network Attack vs Vanilla Split Learning")
    print("═" * 65)
    print(f"  Device        : {device}")
    print(f"  Cut layer     : {args.cut_layer}")
    print(f"  SL epochs     : {args.sl_epochs}")
    print(f"  Attack epochs : {args.attack_epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Save dir      : {args.save_dir}")
    print("═" * 65)

    results = {}

    # ── Data ──────────────────────────────────────────────────────────────────
    _print_section("Loading CIFAR-10")
    train_loader, test_loader = get_dataloader("cifar10", batch_size=args.batch_size)
    print(f"  Train : {len(train_loader.dataset):,} images")
    print(f"  Test  : {len(test_loader.dataset):,} images")

    # ── Step 1: Vanilla Split Learning ────────────────────────────────────────
    _print_section("Step 1 — Vanilla Split Learning (Honest Training)")

    client_model, server_model = create_split_simple_cnn(
        cut_layer=args.cut_layer, num_classes=10
    )

    if args.skip_sl_training and os.path.exists(ckpt_sl):
        print(f"  Loading checkpoint: {ckpt_sl}")
        ckpt = torch.load(ckpt_sl, map_location=device)
        client_model.load_state_dict(ckpt["client_model"])
        server_model.load_state_dict(ckpt["server_model"])
        sl_history = ckpt.get("history", {})
        best_acc = max(sl_history.get("test_acc", [0.0]))
        print(f"  Loaded. Best test accuracy from history: {best_acc*100:.2f}%")
    else:
        print(f"  Training split learning for {args.sl_epochs} epochs …")
        trainer = VanillaSplitTrainer(
            client_model=client_model,
            server_model=server_model,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=args.lr,
            device=device,
        )
        sl_history = trainer.train(epochs=args.sl_epochs)
        trainer.save_checkpoint(ckpt_sl)
        best_acc = max(sl_history["test_acc"])

    results["split_learning"] = {
        "best_test_acc": float(best_acc),
        "cut_layer":     args.cut_layer,
        "sl_epochs":     args.sl_epochs,
    }

    print(f"\n  ✓ Split Learning complete. Best accuracy: {best_acc*100:.2f}%")
    print(f"    (The server saw every smashed activation during training.)")

    # ── Step 2: Build Smashed Dataset ─────────────────────────────────────────
    _print_section("Step 2 — Server Builds Smashed Dataset")
    print("  The server runs auxiliary data through the trained client model")
    print("  to collect (smashed, original) pairs for attack training …")

    attack = InverseNetworkAttack(
        client_model=client_model,
        cut_layer=args.cut_layer,
        lr=args.attack_lr,
        device=device,
    )

    # Server uses the training set as auxiliary data (it observed these smashed
    # activations during the honest SL training phase)
    smashed_dataset = attack.build_smashed_dataset(
        data_loader=train_loader,
        max_samples=None,
    )

    print(f"\n  ✓ Collected {len(smashed_dataset):,} (smashed, original) pairs")

    # ── Step 3: Train Inverse Network ─────────────────────────────────────────
    _print_section("Step 3 — Train Inverse Network (Post-hoc Attack)")
    print("  Training decoder: smashed data → reconstructed image")
    print("  This is a purely offline step — no further interaction with client …\n")

    attack_loader_kwargs = dict(
        batch_size=args.attack_batch_size,
    )

    attack_history = attack.train(
        train_dataset=smashed_dataset,
        val_loader=test_loader,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        verbose=True,
    )

    attack.save(ckpt_atk)

    # ── Step 4: Full Evaluation ────────────────────────────────────────────────
    _print_section("Step 4 — Reconstruction Evaluation")
    print("  Computing SSIM, PSNR, and distance correlation on test set …\n")

    eval_metrics = attack.evaluate_full(
        data_loader=test_loader,
        n_report_batches=8,
    )

    results["attack"] = {
        "ssim":        eval_metrics["ssim"],
        "psnr":        eval_metrics["psnr"],
        "dcor":        eval_metrics.get("dcor", float("nan")),
        "attack_epochs": args.attack_epochs,
    }

    # ── Summary table ──────────────────────────────────────────────────────────
    _print_section("Summary")

    col = 18
    fmt = f"  {{:<{col}}} {{:>{col}}} {{:>{col}}} {{:>{col}}} {{:>{col}}}"
    div = "  " + "─" * (col * 5 + 4)

    print(fmt.format("Condition", "Test Acc", "SSIM ↑", "PSNR (dB) ↑", "dCor ↓"))
    print(div)
    print(fmt.format(
        f"Vanilla SL (cut={args.cut_layer})",
        f"{best_acc*100:.2f}%",
        f"{eval_metrics['ssim']:.4f}",
        f"{eval_metrics['psnr']:.2f}",
        f"{eval_metrics.get('dcor', float('nan')):.4f}",
    ))
    print(div)

    print(f"""
  Metric guide:
    SSIM > 0.30  → attack recovers low-level structure  (privacy concern)
    SSIM > 0.50  → attack recovers recognisable content  (significant breach)
    SSIM > 0.70  → near-faithful reconstruction          (critical breach)
    dCor > 0.50  → strong statistical dependency between smashed data and input
                   (validates motivation for novel defenses)
""")

    # Training curves (last 30 epochs)
    best_attack_ssim = max(attack_history.get("val_ssim", [0.0]) or [0.0])
    best_attack_psnr = max(attack_history.get("val_psnr", [0.0]) or [0.0])
    print(f"  Best val SSIM during attack training : {best_attack_ssim:.4f}")
    print(f"  Best val PSNR during attack training : {best_attack_psnr:.2f} dB")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    results["config"] = {
        "cut_layer":     args.cut_layer,
        "sl_epochs":     args.sl_epochs,
        "attack_epochs": args.attack_epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "attack_lr":     args.attack_lr,
        "device":        device,
    }

    # Include val curves (truncated to last 30 entries for readability)
    n = 30
    results["attack_training_curves"] = {
        "val_ssim": attack_history.get("val_ssim", [])[-n:],
        "val_psnr": attack_history.get("val_psnr", [])[-n:],
        "train_loss": attack_history.get("train_loss", [])[-n:],
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_section("Output Files")
    print(f"  SL checkpoint      → {ckpt_sl}")
    print(f"  Attack checkpoint  → {ckpt_atk}")
    print(f"  Results JSON       → {json_path}")

    print("\n" + "═" * 65)
    print("  Inverse Network Attack Test Complete")
    print("═" * 65)
    print(f"  SL test accuracy     : {best_acc*100:.2f}%")
    print(f"  Reconstruction SSIM  : {eval_metrics['ssim']:.4f}")
    print(f"  Reconstruction PSNR  : {eval_metrics['psnr']:.2f} dB")
    print(f"  Distance Correlation : {eval_metrics.get('dcor', float('nan')):.4f}")
    print("═" * 65)
    print()
    print("  → These results serve as the BASELINE for comparing NoPeekNN")
    print("    and future defenses in evaluate_attack_defense.py")
    print()


if __name__ == "__main__":
    main()