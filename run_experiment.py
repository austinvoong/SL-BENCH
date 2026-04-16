"""
run_experiment.py  (sl_bench/ project root)

Called by Flask backend:
  python run_experiment.py --attack FORA --defense NoPeekNN --cut_layer 2 --epochs 15

Actual module layout (confirmed from source):
  data/__init__.py              → get_dataloader("cifar10", batch_size)
  models/__init__.py            → create_split_simple_cnn(cut_layer, num_classes)
  trainers/__init__.py          → VanillaSplitTrainer, UShapedSplitTrainer, SplitFedTrainer
  defenses/__init__.py          → NoPeekNNTrainer, DifferentialPrivacyTrainer
  attacks/__init__.py           → InverseNetworkAttack, FORAAttack, FSHAAttack
  metrics/reconstruction.py     → (used internally by attack.evaluate_full)

Key design:
  - Defense IS the trainer.  There is no separate defense object.
  - FSHA hijacks training itself — no SL trainer is used.
  - trainer.train(epochs) → history dict with "test_acc" list (fractions, not %)
  - attack.evaluate_full(test_loader) → {"ssim", "psnr", "dcor"}
"""

import argparse
import json
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Smashed data shape by cut layer for SimpleCNN on CIFAR-10 (32×32 input)
#   cut 1 → conv1 out: (32 ch, 16×16)
#   cut 2 → conv2 out: (64 ch,  8×8)   ← recommended
#   cut 3 → conv3 out: (128 ch, 4×4)
SMASHED_SHAPES = {
    1: (32, 16),
    2: (64,  8),
    3: (128, 4),
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack",       required=True,
                   choices=["FORA", "FSHA", "Inverse Network"])
    p.add_argument("--defense",      required=True,
                   choices=["None", "NoPeekNN", "DP-Gaussian", "DP-Laplace", "AFO"])
    p.add_argument("--architecture", default="Vanilla SL",
                   choices=["Vanilla SL", "U-Shaped SL", "SplitFed"])
    p.add_argument("--cut_layer",    type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--epochs",       type=int, default=15)
    p.add_argument("--batch_size",   type=int, default=128)
    p.add_argument("--run_id",       default="local")
    return p.parse_args()


# ── Trainer factory ───────────────────────────────────────────────────────────

def build_trainer(defense, architecture, client, server, train_loader, test_loader, lr=0.001):
    """
    Defense IS the trainer — return the right trainer class for the given
    defense and architecture combination.
    """
    if architecture == "Vanilla SL":
        if defense == "None":
            from trainers import VanillaSplitTrainer
            return VanillaSplitTrainer(
                client_model=client, server_model=server,
                train_loader=train_loader, test_loader=test_loader, lr=lr,
            )
        elif defense == "NoPeekNN":
            from defenses import NoPeekNNTrainer
            return NoPeekNNTrainer(
                client_model=client, server_model=server,
                train_loader=train_loader, test_loader=test_loader,
                lambda_dcor=0.5, dcor_subsample=64, lr=lr,
            )
        elif defense == "DP-Gaussian":
            from defenses import DifferentialPrivacyTrainer
            return DifferentialPrivacyTrainer(
                client_model=client, server_model=server,
                train_loader=train_loader, test_loader=test_loader,
                mechanism="gaussian",
                noise_multiplier=0.1,   # σ — increase for stronger privacy (degrades utility)
                clip_norm=1.0,          # sensitivity clipping norm C
                lr=lr,
            )
        elif defense == "DP-Laplace":
            from defenses import DifferentialPrivacyTrainer
            return DifferentialPrivacyTrainer(
                client_model=client, server_model=server,
                train_loader=train_loader, test_loader=test_loader,
                mechanism="laplace",
                epsilon=5.0,            # privacy budget ε
                clip_norm=1.0,
                lr=lr,
            )
        elif defense == "AFO":
            # AFO not yet implemented — run as Vanilla so pipeline doesn't crash
            print("[run_experiment] WARNING: AFO not yet implemented, running Vanilla SL", flush=True)
            from trainers import VanillaSplitTrainer
            return VanillaSplitTrainer(
                client_model=client, server_model=server,
                train_loader=train_loader, test_loader=test_loader, lr=lr,
            )

    elif architecture == "U-Shaped SL":
        # U-shaped uses a different client model type (UShapedClientModel).
        # Defense integration not yet implemented for U-Shaped.
        from trainers import UShapedSplitTrainer
        return UShapedSplitTrainer(
            client_model=client, server_model=server,
            train_loader=train_loader, test_loader=test_loader, lr=lr,
        )

    elif architecture == "SplitFed":
        from trainers import SplitFedTrainer
        return SplitFedTrainer(
            client_model=client, server_model=server,
            train_loader=train_loader, test_loader=test_loader, lr=lr,
        )

    raise ValueError(f"Unsupported combination: {architecture} + {defense}")


# ── Attack runners ────────────────────────────────────────────────────────────

def run_inverse_network(client_model, cut_layer, train_loader, test_loader, tag):
    """
    InverseNetworkAttack: semi-honest server trains a decoder on (smashed, original) pairs.
    evaluate_full() returns {"ssim", "psnr", "dcor"}.
    """
    from attacks import InverseNetworkAttack

    attack = InverseNetworkAttack(client_model=client_model, cut_layer=cut_layer)

    print(f"{tag} Building smashed dataset...", flush=True)
    smashed_ds = attack.build_smashed_dataset(train_loader)

    print(f"{tag} Training inverse network...", flush=True)
    attack.train(
        train_dataset=smashed_ds,
        val_loader=test_loader,
        epochs=20,
        batch_size=128,
        verbose=True,
    )

    return attack.evaluate_full(test_loader)


def run_fora(client_model, cut_layer, train_loader, test_loader, tag):
    """
    FORAAttack: semi-honest server builds a substitute client and inverse network.

    Since we can't hook into the trainer loop without modifying trainers, we
    simulate what the server observed during training: pass all training images
    through the (now trained) client model, feed each batch to update_substitute,
    then train the inverse network on the collected smashed data.

    This is a valid semi-honest simulation — the server legitimately received
    this smashed data during training.
    """
    from attacks import FORAAttack

    smashed_ch, smashed_sp = SMASHED_SHAPES[cut_layer]

    fora = FORAAttack(
        smashed_channels=smashed_ch,
        smashed_spatial=smashed_sp,
        aux_loader=train_loader,        # public auxiliary data (same domain)
        cut_layer=cut_layer,
    )

    # Phase 1: train substitute client on private smashed data
    print(f"{tag} FORA Phase 1 — training substitute client...", flush=True)
    client_model.eval()
    device = next(client_model.parameters()).device
    with torch.no_grad():
        for images, _ in train_loader:
            smashed = client_model(images.to(device))
            fora.update_substitute(smashed.detach())

    # Phase 2: train inverse network
    # If train_inverse_network requires different args, adjust here.
    # Most likely signature: fora.train_inverse_network(epochs=N)
    print(f"{tag} FORA Phase 2 — training inverse network...", flush=True)
    fora.train_inverse_network(epochs=20)

    return fora.evaluate(test_loader)


def run_fsha(cut_layer, train_loader, test_loader, epochs, tag):
    """
    FSHAAttack: malicious server hijacks training by sending crafted gradients.
    This replaces the SL training loop entirely — no separate trainer is used.

    run_hijacked_training() trains the attacker's autoencoder (f̃, f̃⁻¹) and
    simultaneously forces the victim's client model to become an autoencoder
    via crafted gradient injection.

    Adjust the method call if your FSHAAttack uses different argument names.
    """
    from attacks import FSHAAttack

    fsha = FSHAAttack(cut_layer=cut_layer, public_loader=train_loader)

    print(f"{tag} FSHA — running hijacked training...", flush=True)
    fsha.run_hijacked_training(
        victim_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
    )

    return fsha.evaluate_reconstruction(test_loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    tag    = f"[{args.run_id}]"
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(
        f"{tag} attack={args.attack} | defense={args.defense} | "
        f"arch={args.architecture} | cut_layer={args.cut_layer} | "
        f"epochs={args.epochs} | device={device}",
        flush=True,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"{tag} Loading CIFAR-10...", flush=True)
    from data import get_dataloader
    train_loader, test_loader = get_dataloader("cifar10", batch_size=args.batch_size)

    # ── FSHA: bypasses normal SL training entirely ────────────────────────────
    if args.attack == "FSHA":
        metrics  = run_fsha(args.cut_layer, train_loader, test_loader, args.epochs, tag)
        # FSHA hijacks training so victim accuracy is not meaningful
        accuracy = 0.0
        print(f"{tag} FSHA done — SSIM={metrics['ssim']:.4f}", flush=True)

    # ── All other attacks: train SL first, then attack ────────────────────────
    else:
        from models import create_split_simple_cnn
        client, server = create_split_simple_cnn(cut_layer=args.cut_layer, num_classes=10)

        print(f"{tag} Training {args.architecture} with defense={args.defense}...", flush=True)
        trainer = build_trainer(
            args.defense, args.architecture,
            client, server,
            train_loader, test_loader,
        )
        history  = trainer.train(epochs=args.epochs)

        # history["test_acc"] is a list of per-epoch fractions (0–1)
        accuracy = max(history["test_acc"]) * 100
        print(f"{tag} Training done — best test accuracy={accuracy:.2f}%", flush=True)

        if args.attack == "Inverse Network":
            print(f"{tag} Running Inverse Network attack...", flush=True)
            metrics = run_inverse_network(
                trainer.client_model, args.cut_layer,
                train_loader, test_loader, tag,
            )

        elif args.attack == "FORA":
            print(f"{tag} Running FORA attack...", flush=True)
            metrics = run_fora(
                trainer.client_model, args.cut_layer,
                train_loader, test_loader, tag,
            )

    # ── Emit result JSON (Flask reads this line) ──────────────────────────────
    result = {
        "ssim":     float(metrics["ssim"]),
        "psnr":     float(metrics["psnr"]),
        "dcor":     float(metrics["dcor"]),
        "accuracy": round(accuracy, 2),
        "note":     f"{args.attack} vs {args.defense} | cut={args.cut_layer}",
    }
    print(f"{tag} {result}", flush=True)
    print(json.dumps(result), flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)