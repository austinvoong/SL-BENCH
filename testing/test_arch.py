"""
testing/run_arch.py

Quick-start script demonstrating all three split learning architectures:

    1. Vanilla Split Learning  (VanillaSplitTrainer)
    2. U-Shaped Split Learning (UShapedSplitTrainer)
    3. SplitFed                (SplitFedTrainer)

Run a specific experiment:
    python testing/run_arch.py --mode vanilla
    python testing/run_arch.py --mode ushaped
    python testing/run_arch.py --mode splitfed

Run all three sequentially:
    python testing/run_arch.py --mode all

"""

import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

import torch

from data import get_dataloader
from models import create_split_simple_cnn, create_ushaped_models
from trainers import SplitFedTrainer, UShapedSplitTrainer, VanillaSplitTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "dataset":     "cifar10",
    "batch_size":  128,
    "num_classes": 10,
    "lr":          0.001,
    "epochs":      10,      # vanilla / u-shaped
    "rounds":      10,      # splitfed
    "num_clients": 3,       # splitfed
    "cut_layer":   2,       # vanilla (1–3)
    "cut_1":       2,       # u-shaped (1 or 2)
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Vanilla Split Learning
# ─────────────────────────────────────────────────────────────────────────────

def run_vanilla():
    print("\n" + "━" * 70)
    print("  EXPERIMENT 1 — Vanilla Split Learning")
    print("━" * 70)

    train_loader, test_loader = get_dataloader(
        CFG["dataset"], batch_size=CFG["batch_size"]
    )

    client, server = create_split_simple_cnn(
        cut_layer=CFG["cut_layer"],
        num_classes=CFG["num_classes"],
    )

    trainer = VanillaSplitTrainer(
        client_model=client,
        server_model=server,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=CFG["lr"],
    )

    history = trainer.train(epochs=CFG["epochs"])
    trainer.save_checkpoint(os.path.join(CHECKPOINTS_DIR, "vanilla_sl.pt"))

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — U-Shaped Split Learning
# ─────────────────────────────────────────────────────────────────────────────

def run_ushaped():
    print("\n" + "━" * 70)
    print("  EXPERIMENT 2 — U-Shaped Split Learning")
    print("  (Labels never leave the client)")
    print("━" * 70)

    train_loader, test_loader = get_dataloader(
        CFG["dataset"], batch_size=CFG["batch_size"]
    )

    client, server = create_ushaped_models(
        cut_1=CFG["cut_1"],
        num_classes=CFG["num_classes"],
    )

    trainer = UShapedSplitTrainer(
        client_model=client,
        server_model=server,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=CFG["lr"],
    )

    history = trainer.train(epochs=CFG["epochs"])
    trainer.save_checkpoint(os.path.join(CHECKPOINTS_DIR, "ushaped_sl.pt"))

    # Show smashed/server output shapes (useful for attack surface analysis)
    sample_batch, _ = next(iter(test_loader))
    smashed    = trainer.get_smashed_data(sample_batch[:4])
    server_out = trainer.get_server_output(sample_batch[:4])
    print(f"\n  Smashed data shape  (client → server): {tuple(smashed.shape)}")
    print(f"  Server output shape (server → client): {tuple(server_out.shape)}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — SplitFed
# ─────────────────────────────────────────────────────────────────────────────

def run_splitfed():
    print("\n" + "━" * 70)
    print("  EXPERIMENT 3 — SplitFed")
    print(f"  ({CFG['num_clients']} clients, FedAvg after each round)")
    print("━" * 70)

    train_loader, test_loader = get_dataloader(
        CFG["dataset"], batch_size=CFG["batch_size"]
    )

    def client_model_fn():
        client, _ = create_split_simple_cnn(
            cut_layer=CFG["cut_layer"],
            num_classes=CFG["num_classes"],
        )
        return client

    _, server = create_split_simple_cnn(
        cut_layer=CFG["cut_layer"],
        num_classes=CFG["num_classes"],
    )

    trainer = SplitFedTrainer(
        client_model_fn=client_model_fn,
        server_model=server,
        train_loader=train_loader,
        test_loader=test_loader,
        num_clients=CFG["num_clients"],
        lr=CFG["lr"],
    )

    history = trainer.train(rounds=CFG["rounds"])
    trainer.save_checkpoint(os.path.join(CHECKPOINTS_DIR, "splitfed.pt"))

    divergence = trainer.get_client_divergence()
    print("\n  Post-round client divergence (should be ~0 after FedAvg):")
    for k, v in divergence.items():
        print(f"    {k}: {v:.6f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="SL-BENCH Architecture Experiments")
    parser.add_argument(
        "--mode",
        choices=["vanilla", "ushaped", "splitfed", "all"],
        default="all",
        help="Which architecture experiment to run",
    )
    args = parser.parse_args()

    if args.mode in ("vanilla", "all"):
        run_vanilla()

    if args.mode in ("ushaped", "all"):
        run_ushaped()

    if args.mode in ("splitfed", "all"):
        run_splitfed()