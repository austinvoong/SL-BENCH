"""
testing/test_nopeeknn.py

Vanilla SL vs NoPeekNN defense
Attack: Inverse Network (semi-honest reconstruction baseline)

Runs two conditions and prints a side-by-side summary:
    A) Vanilla SL   → Inverse Network Attack
    B) NoPeekNN SL  → Inverse Network Attack

Usage:
    python testing/test_nopeeknn.py
    python testing/test_nopeeknn.py --sl-epochs 10 --attack-epochs 20 --lambda-dcor 0.5
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
from defenses import NoPeekNNTrainer
from attacks import InverseNetworkAttack


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def header(title: str):
    bar = "═" * 60
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
    save_path = os.path.join(args.save_dir, f"inv_net_{label.lower().replace(' ', '_')}.pt")
    attack.save(save_path)
    return attack.evaluate_full(test_loader)


def print_summary(results: dict, args):
    header("RESULTS SUMMARY  —  NoPeekNN vs Vanilla")

    col = 18
    fmt = f"  {{:<{col}}} {{:>10}} {{:>10}} {{:>12}} {{:>10}}"
    div = "  " + "─" * (col + 46)

    print(fmt.format("Defense", "Test Acc", "SSIM ↑", "PSNR (dB) ↑", "dCor ↓"))
    print(div)

    labels = {
        "vanilla": "None (Baseline)",
        "nopeeknn": f"NoPeekNN λ={args.lambda_dcor}",
    }
    for key, res in results.items():
        print(fmt.format(
            labels[key],
            f"{res['test_accuracy']*100:.2f}%",
            f"{res['ssim']:.4f}",
            f"{res['psnr']:.2f}",
            f"{res['dcor']:.4f}",
        ))
    print(div)

    if "vanilla" in results and "nopeeknn" in results:
        va, np_ = results["vanilla"], results["nopeeknn"]
        acc_cost = (va["test_accuracy"] - np_["test_accuracy"]) * 100
        within   = "✓ within target" if abs(acc_cost) < 5 else "✗ exceeds 5pp target"
        print(f"\n  NoPeekNN effect:")
        print(f"    Accuracy cost   : {acc_cost:+.2f}pp  ({within})")
        print(f"    SSIM reduction  : {va['ssim'] - np_['ssim']:+.4f}  (positive = attacker reconstructs worse)")
        print(f"    PSNR reduction  : {va['psnr'] - np_['psnr']:+.2f} dB")
        print(f"    dCor reduction  : {va['dcor'] - np_['dcor']:+.4f}  (positive = less leakage)")
        print()
        if va["ssim"] - np_["ssim"] < 0.02:
            print("  ⚠  SSIM reduction < 0.02 — NoPeekNN provides marginal protection.")
            print("     This is expected: FORA-class attacks bypass statistical independence.")
            print("     Motivates the need for AFO and other novel defenses.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    results = {}

    train_loader, test_loader = get_dataloader("cifar10", batch_size=args.batch_size)

    # ── Condition A: Vanilla SL ───────────────────────────────────────────────
    if not args.skip_vanilla:
        header("CONDITION A — Vanilla Split Learning (No Defense)")
        client_a, server_a = create_split_simple_cnn(cut_layer=args.cut_layer, num_classes=10)
        trainer_a = VanillaSplitTrainer(
            client_model=client_a, server_model=server_a,
            train_loader=train_loader, test_loader=test_loader, lr=args.lr,
        )
        history_a = trainer_a.train(epochs=args.sl_epochs)
        trainer_a.save_checkpoint(os.path.join(args.save_dir, "vanilla_sl.pt"))

        metrics_a = run_attack(
            trainer_a.client_model, train_loader, test_loader,
            args.cut_layer, args, "Vanilla SL",
        )
        results["vanilla"] = {"test_accuracy": max(history_a["test_acc"]), **metrics_a}

    # ── Condition B: NoPeekNN ─────────────────────────────────────────────────
    header(f"CONDITION B — NoPeekNN Defense  (λ = {args.lambda_dcor})")
    client_b, server_b = create_split_simple_cnn(cut_layer=args.cut_layer, num_classes=10)
    trainer_b = NoPeekNNTrainer(
        client_model=client_b, server_model=server_b,
        train_loader=train_loader, test_loader=test_loader,
        lambda_dcor=args.lambda_dcor, dcor_subsample=args.dcor_subsample, lr=args.lr,
    )
    history_b = trainer_b.train(epochs=args.sl_epochs)
    trainer_b.save_checkpoint(os.path.join(args.save_dir, "nopeeknn.pt"))

    metrics_b = run_attack(
        trainer_b.client_model, train_loader, test_loader,
        args.cut_layer, args, "NoPeekNN",
    )
    results["nopeeknn"] = {"test_accuracy": max(history_b["test_acc"]), **metrics_b}

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(results, args)

    out_path = os.path.join(args.save_dir, "nopeeknn_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SL-BENCH: NoPeekNN vs Vanilla test")

    # SL training
    parser.add_argument("--sl-epochs",        type=int,   default=15)
    parser.add_argument("--cut-layer",         type=int,   default=2)
    parser.add_argument("--batch-size",        type=int,   default=128)
    parser.add_argument("--lr",                type=float, default=0.001)

    # NoPeekNN
    parser.add_argument("--lambda-dcor",       type=float, default=0.5)
    parser.add_argument("--dcor-subsample",    type=int,   default=64)

    # Attack
    parser.add_argument("--attack-epochs",     type=int,   default=30)
    parser.add_argument("--attack-batch-size", type=int,   default=128)

    # Control
    parser.add_argument("--skip-vanilla",      action="store_true",
                        help="Skip vanilla condition (reuse existing checkpoint)")
    parser.add_argument("--save-dir",          type=str,
                        default=os.path.join(ROOT_DIR, "checkpoints", "test_nopeeknn"))

    main(parser.parse_args())