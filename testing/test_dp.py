"""
testing/test_dp.py

Vanilla SL vs Differential Privacy defense
Attack: Inverse Network (semi-honest reconstruction baseline)

Tests multiple DP configurations against the vanilla baseline:
    A) Vanilla SL                      → Inverse Network Attack
    B) DP Gaussian  σ/C=0.01  (weak)   → Inverse Network Attack
    C) DP Gaussian  σ/C=0.1   (mid)    → Inverse Network Attack
    D) DP Gaussian  σ/C=1.0   (strong) → Inverse Network Attack
    E) DP Laplace   ε=1.0              → Inverse Network Attack

Run all or pick specific conditions:
    python testing/test_dp.py
    python testing/test_dp.py --conditions vanilla dp_gaussian_mid dp_laplace
    python testing/test_dp.py --sl-epochs 10 --attack-epochs 20 --noise-multiplier 0.1
"""

import os
import sys
import json
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch

from data import get_dataloader
from models import create_split_simple_cnn
from trainers import VanillaSplitTrainer
from defenses import DifferentialPrivacyTrainer
from attacks import InverseNetworkAttack


# ─────────────────────────────────────────────────────────────────────────────
# Condition registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (display_label, mechanism, noise_multiplier, epsilon)
DP_CONDITIONS = {
    "dp_gaussian_weak":   ("DP-G σ/C=0.01",  "gaussian", 0.01, None),
    "dp_gaussian_mid":    ("DP-G σ/C=0.1",   "gaussian", 0.10, None),
    "dp_gaussian_strong": ("DP-G σ/C=1.0",   "gaussian", 1.00, None),
    "dp_laplace":         ("DP-L ε=1.0",     "laplace",  None, 1.0),
}
ALL_CONDITIONS = ["vanilla"] + list(DP_CONDITIONS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def header(title: str):
    bar = "═" * 65
    print(f"\n{bar}\n  {title}\n{bar}")


def run_attack(client_model, train_loader, test_loader, cut_layer, args, label) -> dict:
    """Build, train, and evaluate the Inverse Network against a trained client model."""
    print(f"\n  ▶ Inverse Network attack on: {label}")
    attack = InverseNetworkAttack(client_model=client_model, cut_layer=cut_layer)
    smashed_ds = attack.build_smashed_dataset(train_loader)
    attack.train(
        train_dataset=smashed_ds,
        val_loader=test_loader,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        verbose=True,
    )
    safe_label = label.lower().replace(" ", "_").replace("/", "").replace("=", "")
    save_path  = os.path.join(args.save_dir, f"inv_net_{safe_label}.pt")
    attack.save(save_path)
    return attack.evaluate_full(test_loader)


# ─────────────────────────────────────────────────────────────────────────────
# Condition runners
# ─────────────────────────────────────────────────────────────────────────────

def run_vanilla(args, train_loader, test_loader) -> dict:
    header("CONDITION A — Vanilla Split Learning (No Defense)")
    client, server = create_split_simple_cnn(cut_layer=args.cut_layer, num_classes=10)
    trainer = VanillaSplitTrainer(
        client_model=client, server_model=server,
        train_loader=train_loader, test_loader=test_loader, lr=args.lr,
    )
    history = trainer.train(epochs=args.sl_epochs)
    trainer.save_checkpoint(os.path.join(args.save_dir, "vanilla_sl.pt"))
    metrics = run_attack(
        trainer.client_model, train_loader, test_loader,
        args.cut_layer, args, "Vanilla SL",
    )
    return {"test_accuracy": max(history["test_acc"]), **metrics}


def run_dp_condition(key, args, train_loader, test_loader) -> dict:
    display_label, mechanism, noise_multiplier, epsilon = DP_CONDITIONS[key]
    header(f"CONDITION — {display_label}  (C={args.clip_norm})")

    client, server = create_split_simple_cnn(cut_layer=args.cut_layer, num_classes=10)

    # Build trainer with appropriate mechanism params
    trainer_kwargs = dict(
        client_model=client, server_model=server,
        train_loader=train_loader, test_loader=test_loader,
        mechanism=mechanism,
        clip_norm=args.clip_norm,
        delta=args.delta,
        lr=args.lr,
    )
    if mechanism == "gaussian":
        trainer_kwargs["noise_multiplier"] = noise_multiplier
    else:
        trainer_kwargs["epsilon"] = epsilon

    trainer = DifferentialPrivacyTrainer(**trainer_kwargs)
    history = trainer.train(epochs=args.sl_epochs)

    # Save checkpoint + privacy budget report
    ckpt_name = f"dp_{key}.pt"
    trainer.save_checkpoint(os.path.join(args.save_dir, ckpt_name))
    n_steps = args.sl_epochs * len(train_loader)
    trainer.privacy_budget_report(n_steps)

    metrics = run_attack(
        trainer.client_model, train_loader, test_loader,
        args.cut_layer, args, display_label,
    )
    return {"test_accuracy": max(history["test_acc"]), **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    header("RESULTS SUMMARY  —  DP Defense vs Vanilla")

    label_map = {"vanilla": "None (Baseline)"}
    for k, (lbl, *_) in DP_CONDITIONS.items():
        label_map[k] = lbl

    col = 20
    fmt = f"  {{:<{col}}} {{:>10}} {{:>10}} {{:>12}} {{:>10}}"
    div = "  " + "─" * (col + 46)

    print(fmt.format("Defense", "Test Acc", "SSIM ↑", "PSNR (dB) ↑", "dCor ↓"))
    print(div)

    for key, res in results.items():
        print(fmt.format(
            label_map.get(key, key),
            f"{res['test_accuracy']*100:.2f}%",
            f"{res['ssim']:.4f}",
            f"{res['psnr']:.2f}",
            f"{res['dcor']:.4f}",
        ))
    print(div)

    if "vanilla" not in results:
        return

    va = results["vanilla"]
    print(f"\n  Effect relative to Vanilla baseline:")
    print(f"  {'Defense':<{col}} {'Acc Δ':>8}  {'SSIM Δ':>10}  {'PSNR Δ':>10}  {'dCor Δ':>10}  {'<5pp?':>6}")
    print("  " + "─" * (col + 52))

    for key, res in results.items():
        if key == "vanilla":
            continue
        acc_delta  = (res["test_accuracy"] - va["test_accuracy"]) * 100
        ssim_delta = va["ssim"] - res["ssim"]
        psnr_delta = va["psnr"] - res["psnr"]
        dcor_delta = va["dcor"] - res["dcor"]
        within     = "✓" if abs(acc_delta) < 5 else "✗"
        print(
            f"  {label_map.get(key, key):<{col}} "
            f"{acc_delta:>+7.2f}pp  "
            f"{ssim_delta:>+10.4f}  "
            f"{psnr_delta:>+10.2f}  "
            f"{dcor_delta:>+10.4f}  "
            f"{within:>6}"
        )

    print()
    print("  Interpretation guide:")
    print("    Acc Δ  < 0   → defense costs accuracy (expected)")
    print("    SSIM Δ > 0   → reconstruction quality degraded (defense working)")
    print("    dCor Δ > 0   → less statistical leakage (defense working)")
    print()
    print("  Note: DP noise disrupts the Inverse Network (baseline attack)")
    print("  but FORA's substitute-client approach is more robust to noise.")
    print("  These results motivate AFO as a more targeted defense.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    results = {}

    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    train_loader, test_loader = get_dataloader("cifar10", batch_size=args.batch_size)

    if "vanilla" in conditions:
        results["vanilla"] = run_vanilla(args, train_loader, test_loader)

    for key in DP_CONDITIONS:
        if key in conditions:
            results[key] = run_dp_condition(key, args, train_loader, test_loader)

    print_summary(results)

    out_path = os.path.join(args.save_dir, "dp_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SL-BENCH: DP Defense vs Vanilla test")

    parser.add_argument(
        "--conditions", nargs="+", choices=ALL_CONDITIONS,
        help="Conditions to run (default: all). "
             "E.g. --conditions vanilla dp_gaussian_mid dp_laplace"
    )

    # SL training
    parser.add_argument("--sl-epochs",        type=int,   default=15)
    parser.add_argument("--cut-layer",         type=int,   default=2)
    parser.add_argument("--batch-size",        type=int,   default=128)
    parser.add_argument("--lr",                type=float, default=0.001)

    # DP shared
    parser.add_argument("--clip-norm",         type=float, default=1.0,
                        help="L2 sensitivity clipping norm C")
    parser.add_argument("--delta",             type=float, default=1e-5,
                        help="δ for Gaussian (ε,δ)-DP")

    # Attack
    parser.add_argument("--attack-epochs",     type=int,   default=30)
    parser.add_argument("--attack-batch-size", type=int,   default=128)

    # I/O
    parser.add_argument("--save-dir",          type=str,
                        default=os.path.join(ROOT_DIR, "checkpoints", "test_dp"))

    main(parser.parse_args())