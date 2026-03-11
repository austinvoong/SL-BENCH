"""
testing/test_fsha.py
Test the Feature-Space Hijacking Attack (FSHA) against Vanilla Split Learning.

Demonstrates that a MALICIOUS SERVER can reconstruct private client training
data with high fidelity by hijacking the client's feature space — without
any deviation visible to the client.

Protocol:
    - The client trains normally (sends smashed data, receives "task gradient")
    - The server actually sends adversarial gradients (∂(-D(smashed))/∂smashed)
    - After N setup iterations, f maps X_priv → Z̃  (attacker's target space)
    - Reconstruction: X̃_priv = f̃⁻¹(f(X_priv))  achieves high SSIM/PSNR

Public data:  CIFAR-10 test set  (server's auxiliary — same domain, no overlap)
Private data: CIFAR-10 train set (client's private data being leaked)

Usage (from project root):
    python testing/test_fsha.py
    python testing/test_fsha.py --n-iters 5000 --cut-layer 2 --eval-interval 500
    python testing/test_fsha.py --device cpu --n-iters 1000   # quick smoke test

Expected results after ~3000 iterations:
    SSIM > 0.40 (meaningful reconstruction — paper shows >0.80 on simpler datasets)
    PSNR > 15 dB
    (CIFAR-10 is harder than MNIST/CelebA used in the paper; more iters may be needed)
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
from attacks import FSHAAttack


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SL-BENCH: Test FSHA against Vanilla Split Learning"
    )
    # Architecture
    p.add_argument("--cut-layer",    type=int,   default=2,
                   help="SimpleCNN cut layer (1–3). Default: 2")
    # Attack config
    p.add_argument("--n-iters",      type=int,   default=3000,
                   help="Number of FSHA setup iterations. Default: 3000")
    p.add_argument("--eval-interval",type=int,   default=500,
                   help="Evaluate reconstruction every N iters. Default: 500")
    p.add_argument("--lr-client",    type=float, default=1e-5,
                   help="Client model LR (paper uses 1e-5). Default: 1e-5")
    p.add_argument("--lr-ae",        type=float, default=1e-4,
                   help="Pilot+inverse LR. Default: 1e-4")
    p.add_argument("--lr-disc",      type=float, default=1e-4,
                   help="Discriminator LR. Default: 1e-4")
    p.add_argument("--lambda-gp",    type=float, default=500.0,
                   help="WGAN-GP gradient penalty weight (paper: 500). Default: 500")
    p.add_argument("--n-disc-steps", type=int,   default=3,
                   help="D updates per iteration. Default: 3")
    # Data / training
    p.add_argument("--batch-size",   type=int,   default=64,
                   help="Batch size for data loaders. Default: 64")
    # I/O
    p.add_argument("--save-dir",     type=str,   default="results/fsha",
                   help="Directory to save results. Default: results/fsha")
    p.add_argument("--device",       type=str,   default=None,
                   help="Compute device (cuda/mps/cpu). Default: auto-detect")
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
    args = parse_args()
    device_str = _resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "═" * 65)
    print("  SL-BENCH  ·  FSHA vs Vanilla Split Learning")
    print("═" * 65)
    print(f"  Device        : {device_str}")
    print(f"  Cut layer     : {args.cut_layer}")
    print(f"  Setup iters   : {args.n_iters}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Save dir      : {args.save_dir}")
    print("═" * 65)

    # ── Data ──────────────────────────────────────────────────────────────────
    # Private data  = CIFAR-10 train  (victim client's dataset)
    # Public data   = CIFAR-10 test   (server's auxiliary — same domain)
    _print_section("Loading Data")
    train_loader, test_loader = get_dataloader("cifar10", batch_size=args.batch_size)

    print(f"  Private (client) : CIFAR-10 train — {len(train_loader.dataset):,} images")
    print(f"  Public  (server) : CIFAR-10 test  — {len(test_loader.dataset):,} images")

    # ── Victim client model ────────────────────────────────────────────────────
    _print_section("Initialising Victim Client Model")
    client_model, _ = create_split_simple_cnn(
        cut_layer=args.cut_layer, num_classes=10
    )
    n_params = sum(p.numel() for p in client_model.parameters())
    print(f"  Architecture : SimpleCNN client (cut_layer={args.cut_layer})")
    print(f"  Parameters   : {n_params:,}")
    print(f"  (Server-side model not needed — FSHA replaces server entirely)")

    # Client optimizer — low LR because adversarial gradients have large magnitude
    client_optimizer = torch.optim.Adam(
        client_model.parameters(), lr=args.lr_client, betas=(0.5, 0.999)
    )

    # ── FSHA attacker ─────────────────────────────────────────────────────────
    _print_section("Building FSHA Attack (Malicious Server)")
    fsha = FSHAAttack(
        cut_layer=args.cut_layer,
        public_loader=test_loader,
        lr_ae=args.lr_ae,
        lr_disc=args.lr_disc,
        lambda_gp=args.lambda_gp,
        n_disc_steps=args.n_disc_steps,
        device=device_str,
    )

    # ── Hijacked training ──────────────────────────────────────────────────────
    _print_section("Running Hijacked Training")
    history = fsha.run_hijacked_training(
        client_model=client_model,
        client_optimizer=client_optimizer,
        private_loader=train_loader,
        n_setup_iters=args.n_iters,
        eval_interval=args.eval_interval,
        eval_loader=test_loader,    # evaluate on test set images
        verbose=True,
    )

    # ── Final evaluation ───────────────────────────────────────────────────────
    _print_section("Final Reconstruction Evaluation")
    final_metrics = fsha.evaluate_reconstruction(
        client_model=client_model,
        data_loader=test_loader,
        n_batches=20,
    )

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  FSHA Reconstruction Quality (Final)     │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  SSIM : {final_metrics['ssim']:.4f}  "
          f"(higher = better reconstruction)  │")
    print(f"  │  PSNR : {final_metrics['psnr']:.2f} dB"
          f"  (higher = better reconstruction)  │")
    print(f"  └─────────────────────────────────────────┘")

    # Summarise reconstruction quality at each eval checkpoint
    if history["ssim"]:
        print(f"\n  Reconstruction progress over setup iterations:")
        print(f"  {'Iteration':<12} {'SSIM':<10} {'PSNR (dB)':<12}")
        print(f"  {'─'*10:<12} {'─'*8:<10} {'─'*10:<12}")
        for itr, ssim, psnr in zip(history["iter"], history["ssim"], history["psnr"]):
            print(f"  {itr:<12,} {ssim:<10.4f} {psnr:<12.2f}")

    # ── Save results ───────────────────────────────────────────────────────────
    _print_section("Saving Results")

    # Attacker networks
    ckpt_path = os.path.join(args.save_dir, "fsha_attacker.pt")
    fsha.save(ckpt_path)

    # Hijacked client model (for downstream attack analysis)
    client_path = os.path.join(args.save_dir, "hijacked_client.pt")
    torch.save(client_model.state_dict(), client_path)
    print(f"  Hijacked client model → {client_path}")

    # JSON results
    results = {
        "config": {
            "cut_layer":     args.cut_layer,
            "n_iters":       args.n_iters,
            "eval_interval": args.eval_interval,
            "batch_size":    args.batch_size,
            "lr_client":     args.lr_client,
            "lr_ae":         args.lr_ae,
            "lr_disc":       args.lr_disc,
            "lambda_gp":     args.lambda_gp,
            "n_disc_steps":  args.n_disc_steps,
            "device":        device_str,
        },
        "final_metrics": final_metrics,
        "reconstruction_progress": {
            "iter": history["iter"],
            "ssim": history["ssim"],
            "psnr": history["psnr"],
        },
        # Sample of loss curves (last 100 entries to keep file small)
        "loss_curves": {
            "ae_loss":   history["ae_loss"][-100:],
            "disc_loss": history["disc_loss"][-100:],
            "adv_loss":  history["adv_loss"][-100:],
        },
    }

    json_path = os.path.join(args.save_dir, "fsha_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON         → {json_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  FSHA Test Complete")
    print("═" * 65)
    print(f"  Cut layer     : {args.cut_layer}")
    print(f"  Setup iters   : {args.n_iters}")
    print(f"  Final SSIM    : {final_metrics['ssim']:.4f}")
    print(f"  Final PSNR    : {final_metrics['psnr']:.2f} dB")
    print()
    print("  Interpretation:")
    print("  SSIM > 0.40  → attack extracting meaningful structure")
    print("  SSIM > 0.60  → attack producing recognisable reconstructions")
    print("  SSIM > 0.80  → near-perfect reconstruction (paper on MNIST/CelebA)")
    print()
    print("  Note: CIFAR-10 is significantly harder than MNIST/CelebA.")
    print("  Increase --n-iters (5000–10000) for stronger results.")
    print("═" * 65)


if __name__ == "__main__":
    main()