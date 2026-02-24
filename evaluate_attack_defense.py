"""
evaluate_attack_defense.py
──────────────────────────
Full attack/defense evaluation pipeline for SL-BENCH.

Runs a structured comparison:

    Condition A: Vanilla SL (no defense)   → train → Inverse Network Attack
    Condition B: NoPeekNN (λ = 0.5)        → train → Inverse Network Attack

Reports per condition:
    - Classification accuracy (utility cost of the defense)
    - SSIM  (structural similarity of reconstructed vs original)
    - PSNR  (peak signal-to-noise ratio of reconstruction)
    - dCor  (distance correlation between smashed data and input)

This directly produces the comparison table from the proposal:

    Defense     | Test Acc | SSIM   | PSNR   | dCor
    ────────────|──────────|──────────────────|──────
    None        |  ~85%    |  high  |  high  | high   ← attacker wins
    NoPeekNN    |  ~82%    |  lower |  lower | lower  ← partial protection

Usage:
    python evaluate_attack_defense.py
    python evaluate_attack_defense.py --sl-epochs 20 --attack-epochs 30 --lambda-dcor 0.5
"""

import argparse
import os
import json

import torch

from data import get_dataloader
from models import create_split_simple_cnn
from trainers import VanillaSplitTrainer
from defenses import NoPeekNNTrainer
from attacks import InverseNetworkAttack
from metrics import reconstruction_report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def header(title: str):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def run_attack_on_model(
    client_model: torch.nn.Module,
    train_loader,
    test_loader,
    cut_layer: int,
    attack_epochs: int,
    attack_batch_size: int,
    label: str,
    save_dir: str,
) -> dict:
    """
    Build, train, and evaluate the Inverse Network attack against a given client model.

    Returns dict of metrics: ssim, psnr, dcor
    """
    print(f"\n  ▶ Launching Inverse Network attack against: {label}")

    attack = InverseNetworkAttack(
        client_model=client_model,
        cut_layer=cut_layer,
    )

    # Collect smashed data from the training set (server sees this during SL)
    smashed_dataset = attack.build_smashed_dataset(train_loader)

    # Train the inverse network
    attack.train(
        train_dataset=smashed_dataset,
        val_loader=test_loader,
        epochs=attack_epochs,
        batch_size=attack_batch_size,
        verbose=True,
    )

    # Save the attack model
    os.makedirs(save_dir, exist_ok=True)
    attack.save(os.path.join(save_dir, f"inverse_net_{label.lower().replace(' ', '_')}.pt"))

    # Full evaluation report
    metrics = attack.evaluate_full(test_loader)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    all_results = {}

    train_loader, test_loader = get_dataloader(
        "cifar10",
        batch_size=args.batch_size,
    )

    # ── Condition A: Vanilla SL (no defense) ──────────────────────────────────
    if not args.skip_vanilla:
        header("CONDITION A — Vanilla Split Learning (No Defense)")

        client_a, server_a = create_split_simple_cnn(
            cut_layer=args.cut_layer,
            num_classes=10,
        )

        trainer_a = VanillaSplitTrainer(
            client_model=client_a,
            server_model=server_a,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=args.lr,
        )
        history_a = trainer_a.train(epochs=args.sl_epochs)
        trainer_a.save_checkpoint(os.path.join(args.save_dir, "vanilla_sl.pt"))

        best_acc_a = max(history_a["test_acc"])
        print(f"\n  Vanilla SL best test accuracy: {best_acc_a*100:.2f}%")

        metrics_a = run_attack_on_model(
            client_model=trainer_a.client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            cut_layer=args.cut_layer,
            attack_epochs=args.attack_epochs,
            attack_batch_size=args.attack_batch_size,
            label="Vanilla SL",
            save_dir=args.save_dir,
        )
        all_results["vanilla"] = {
            "test_accuracy": best_acc_a,
            **metrics_a,
        }

    # ── Condition B: NoPeekNN ─────────────────────────────────────────────────
    header(f"CONDITION B — NoPeekNN Defense  (λ = {args.lambda_dcor})")

    client_b, server_b = create_split_simple_cnn(
        cut_layer=args.cut_layer,
        num_classes=10,
    )

    trainer_b = NoPeekNNTrainer(
        client_model=client_b,
        server_model=server_b,
        train_loader=train_loader,
        test_loader=test_loader,
        lambda_dcor=args.lambda_dcor,
        dcor_subsample=args.dcor_subsample,
        lr=args.lr,
    )
    history_b = trainer_b.train(epochs=args.sl_epochs)
    trainer_b.save_checkpoint(os.path.join(args.save_dir, "nopeeknn.pt"))

    best_acc_b = max(history_b["test_acc"])
    print(f"\n  NoPeekNN best test accuracy: {best_acc_b*100:.2f}%")

    metrics_b = run_attack_on_model(
        client_model=trainer_b.client_model,
        train_loader=train_loader,
        test_loader=test_loader,
        cut_layer=args.cut_layer,
        attack_epochs=args.attack_epochs,
        attack_batch_size=args.attack_batch_size,
        label="NoPeekNN",
        save_dir=args.save_dir,
    )
    all_results["nopeeknn"] = {
        "test_accuracy": best_acc_b,
        **metrics_b,
    }

    # ── Summary comparison table ───────────────────────────────────────────────
    header("EVALUATION SUMMARY")

    col_w = 14
    row_fmt = f"  {{:<{col_w}}} {{:>{col_w}}} {{:>{col_w}}} {{:>{col_w}}} {{:>{col_w}}}"
    divider = "  " + "─" * (col_w * 5 + 4)

    print(row_fmt.format("Defense", "Test Acc", "SSIM ↑", "PSNR (dB) ↑", "dCor ↓"))
    print(divider)

    labels_map = {
        "vanilla":   "None (Baseline)",
        "nopeeknn": f"NoPeekNN (λ={args.lambda_dcor})",
    }

    for key, res in all_results.items():
        acc  = f"{res['test_accuracy']*100:.2f}%"
        ssim = f"{res['ssim']:.4f}"
        psnr = f"{res['psnr']:.2f}"
        dcor = f"{res['dcor']:.4f}"
        print(row_fmt.format(labels_map[key], acc, ssim, psnr, dcor))

    print(divider)
    print()

    if "vanilla" in all_results and "nopeeknn" in all_results:
        va = all_results["vanilla"]
        nb = all_results["nopeeknn"]
        acc_cost   = (va["test_accuracy"] - nb["test_accuracy"]) * 100
        ssim_delta = va["ssim"]  - nb["ssim"]
        psnr_delta = va["psnr"]  - nb["psnr"]
        dcor_delta = va["dcor"]  - nb["dcor"]

        print("  NoPeekNN Effect:")
        print(f"    Accuracy cost        : {acc_cost:+.2f}pp  ({'within' if abs(acc_cost) < 5 else 'exceeds'} 5% target)")
        print(f"    SSIM reduction       : {ssim_delta:+.4f}  (positive = harder for attacker)")
        print(f"    PSNR reduction       : {psnr_delta:+.2f} dB")
        print(f"    dCor reduction       : {dcor_delta:+.4f}  (positive = less leakage)")
        print()

    # Save results JSON
    results_path = os.path.join(args.save_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SL-BENCH: Attack/Defense Evaluation")

    # Split learning
    parser.add_argument("--sl-epochs",      type=int,   default=15,   help="SL training epochs")
    parser.add_argument("--cut-layer",       type=int,   default=2,    help="SimpleCNN cut layer (1-3)")
    parser.add_argument("--batch-size",      type=int,   default=128,  help="Training batch size")
    parser.add_argument("--lr",              type=float, default=0.001,help="Learning rate")

    # NoPeekNN defense
    parser.add_argument("--lambda-dcor",     type=float, default=0.5,  help="dCor regularization weight")
    parser.add_argument("--dcor-subsample",  type=int,   default=64,   help="Batch subsample for dCor")

    # Inverse network attack
    parser.add_argument("--attack-epochs",   type=int,   default=30,   help="Inverse network training epochs")
    parser.add_argument("--attack-batch-size",type=int,  default=128,  help="Attack training batch size")

    # Control
    parser.add_argument("--skip-vanilla",    action="store_true",      help="Skip vanilla SL condition")
    parser.add_argument("--save-dir",        type=str,   default="checkpoints/eval", help="Output directory")

    args = parser.parse_args()
    main(args)