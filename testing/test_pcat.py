"""
testing/test_pcat.py

Integration test for PCAT against vanilla split learning on CIFAR-10.

PCAT has three phases:
  1. Train a pseudo-client through the current server model during SL.
  2. Train a reverse mapping from pseudo-client smashed data to images.
  3. Reconstruct private inputs from victim smashed data.

Usage:
    python testing/test_pcat.py --cut_layer 2 --sl_epochs 15 --inverse_epochs 30
    python testing/test_pcat.py --quick
"""

import argparse
import json
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

from attacks.inverse_network import InverseNetworkAttack
from attacks.pcat import PCATAttack
from metrics.reconstruction import compute_psnr, compute_ssim
from models.simple_cnn import create_split_simple_cnn


SMASHED_SHAPE = {
    1: (32, 16, 16),
    2: (64, 8, 8),
    3: (128, 4, 4),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test PCAT attack against vanilla split learning"
    )
    parser.add_argument("--cut_layer", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--sl_epochs", type=int, default=15)
    parser.add_argument("--inverse_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--aux_frac", type=float, default=0.1)
    parser.add_argument("--lambda_moment", type=float, default=0.05)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/pcat")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compare_baseline", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 SL epochs, 5 inverse epochs, 500 private samples.",
    )
    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cifar10(data_dir: str):
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
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )
    return train_dataset, test_dataset


def make_loaders(train_dataset, test_dataset, batch_size, aux_frac, train_n=None):
    if train_n is not None:
        train_dataset = Subset(train_dataset, range(train_n))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    n_test = len(test_dataset)
    n_aux = max(int(n_test * aux_frac), 256)
    n_aux = min(n_aux, n_test - 1)
    n_eval = n_test - n_aux

    aux_subset, eval_subset = random_split(
        test_dataset,
        [n_aux, n_eval],
        generator=torch.Generator().manual_seed(42),
    )

    aux_loader = DataLoader(
        aux_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("\nData split:")
    print(f"  Private training  : {len(train_dataset):,} images")
    print(f"  Auxiliary (server): {n_aux:,} images")
    print(f"  Evaluation        : {n_eval:,} images")
    return train_loader, aux_loader, eval_loader


def train_sl_with_pcat(
    client_model: nn.Module,
    server_model: nn.Module,
    train_loader: DataLoader,
    pcat: PCATAttack,
    epochs: int,
    device: torch.device,
):
    criterion = nn.CrossEntropyLoss()
    client_opt = torch.optim.Adam(client_model.parameters(), lr=1e-3)
    server_opt = torch.optim.Adam(server_model.parameters(), lr=1e-3)

    history = {
        "sl_train_loss": [],
        "sl_train_acc": [],
        "pcat_loss": [],
        "pcat_ce": [],
        "pcat_moment": [],
        "pcat_acc": [],
    }

    print(f"\n{'=' * 65}")
    print("  Vanilla SL Training + PCAT Phase 1")
    print(f"{'=' * 65}")
    print(f"  SL epochs      : {epochs}")
    print(f"  PCAT lambda_m  : {pcat.lambda_moment}")
    print(f"{'=' * 65}\n")

    last_epoch = epochs - 1

    for epoch in range(epochs):
        client_model.train()
        server_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        pcat_losses = []
        pcat_ces = []
        pcat_moments = []
        pcat_accs = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            client_opt.zero_grad()
            smashed = client_model(images)

            pcat_metrics = pcat.update_pseudo_client(
                server_model=server_model,
                private_labels=labels.detach(),
                smashed_priv=smashed.detach(),
            )
            pcat_losses.append(pcat_metrics["pseudo_loss"])
            pcat_ces.append(pcat_metrics["pseudo_ce"])
            pcat_moments.append(pcat_metrics["pseudo_moment"])
            pcat_accs.append(pcat_metrics["pseudo_acc"])

            if epoch == last_epoch:
                pcat.add_to_snapshot(smashed.detach())

            smashed_server = smashed.detach().requires_grad_(True)
            server_opt.zero_grad()
            outputs = server_model(smashed_server)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_to_client = smashed_server.grad.clone()
            server_opt.step()

            smashed.backward(grad_to_client)
            client_opt.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        avg_pcat_loss = sum(pcat_losses) / max(len(pcat_losses), 1)
        avg_pcat_ce = sum(pcat_ces) / max(len(pcat_ces), 1)
        avg_pcat_moment = sum(pcat_moments) / max(len(pcat_moments), 1)
        avg_pcat_acc = sum(pcat_accs) / max(len(pcat_accs), 1)

        history["sl_train_loss"].append(avg_loss)
        history["sl_train_acc"].append(avg_acc)
        history["pcat_loss"].append(avg_pcat_loss)
        history["pcat_ce"].append(avg_pcat_ce)
        history["pcat_moment"].append(avg_pcat_moment)
        history["pcat_acc"].append(avg_pcat_acc)

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"SL Loss: {avg_loss:.4f} | "
            f"Train Acc: {avg_acc * 100:5.2f}% | "
            f"PCAT Loss: {avg_pcat_loss:.4f} | "
            f"Pseudo Acc: {avg_pcat_acc * 100:5.2f}% | "
            f"Moment: {avg_pcat_moment:.4f}"
        )

    return history


@torch.no_grad()
def evaluate_sl_accuracy(client_model, server_model, loader, device):
    client_model.eval()
    server_model.eval()

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = server_model(client_model(images))
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def eval_reconstruction(client_model, reconstruct_fn, eval_loader, device, n_batches):
    client_model.eval()
    all_ssim = []
    all_psnr = []

    for batch_idx, (images, _) in enumerate(eval_loader):
        if batch_idx >= n_batches:
            break
        images = images.to(device)
        smashed = client_model(images)
        recon = reconstruct_fn(smashed).clamp(-1, 1)
        all_ssim.append(compute_ssim(recon, images).item())
        all_psnr.append(compute_psnr(recon, images).item())

    return {
        "ssim": sum(all_ssim) / max(len(all_ssim), 1),
        "psnr": sum(all_psnr) / max(len(all_psnr), 1),
    }


def run_baseline_inverse(client_model, train_loader, eval_loader, cut_layer, device, epochs):
    print(f"\n{'-' * 65}")
    print("  Baseline: Inverse Network Attack")
    print(f"{'-' * 65}")

    attack = InverseNetworkAttack(
        client_model=client_model,
        cut_layer=cut_layer,
        device=str(device),
    )
    smashed_dataset = attack.build_smashed_dataset(train_loader, max_samples=5000)
    attack.train(
        train_dataset=smashed_dataset,
        val_loader=eval_loader,
        epochs=epochs,
        verbose=False,
    )
    ssim, psnr = attack._evaluate(eval_loader)
    return {"ssim": ssim, "psnr": psnr}


def main():
    args = parse_args()

    if args.quick:
        print("\n[QUICK MODE] sl_epochs=2, inverse_epochs=5, train_n=500, eval_batches=3")
        args.sl_epochs = 2
        args.inverse_epochs = 5
        args.eval_batches = 3
        train_n = 500
    else:
        train_n = None

    device = get_device(args.device)
    smashed_ch, smashed_h, smashed_w = SMASHED_SHAPE[args.cut_layer]
    assert smashed_h == smashed_w

    print(f"\n{'=' * 65}")
    print("  PCAT Attack Test - Vanilla Split Learning on CIFAR-10")
    print(f"{'=' * 65}")
    print(f"  Device       : {device}")
    print(f"  Cut layer    : {args.cut_layer} ({SMASHED_SHAPE[args.cut_layer]})")
    print(f"  SL epochs    : {args.sl_epochs}")
    print(f"  Inverse ep   : {args.inverse_epochs}")
    print(f"  Lambda moment: {args.lambda_moment}")
    print(f"{'=' * 65}")

    train_dataset, test_dataset = load_cifar10(args.data_dir)
    train_loader, aux_loader, eval_loader = make_loaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        aux_frac=args.aux_frac,
        train_n=train_n,
    )

    client_model, server_model = create_split_simple_cnn(
        cut_layer=args.cut_layer,
        num_classes=10,
    )
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    pcat = PCATAttack(
        smashed_channels=smashed_ch,
        smashed_spatial=smashed_h,
        aux_loader=aux_loader,
        cut_layer=args.cut_layer,
        lambda_moment=args.lambda_moment,
        device=str(device),
    )

    t_start = time.time()
    sl_history = train_sl_with_pcat(
        client_model=client_model,
        server_model=server_model,
        train_loader=train_loader,
        pcat=pcat,
        epochs=args.sl_epochs,
        device=device,
    )
    t_sl = time.time() - t_start

    test_acc = evaluate_sl_accuracy(client_model, server_model, eval_loader, device)
    print(f"\n  SL training complete in {t_sl:.1f}s")
    print(f"  Final SL test accuracy: {test_acc * 100:.2f}%")

    functionality = pcat.measure_functionality(server_model, eval_loader, n_batches=5)
    print(
        f"  Stolen functionality accuracy: "
        f"{functionality['functionality_acc'] * 100:.2f}%"
    )

    t_inv_start = time.time()
    pcat.train_inverse_network(epochs=args.inverse_epochs, verbose=True)
    t_inv = time.time() - t_inv_start
    print(f"  Reverse mapping training: {t_inv:.1f}s")

    pcat_results = eval_reconstruction(
        client_model=client_model,
        reconstruct_fn=lambda z: pcat.reconstruct_batch(z),
        eval_loader=eval_loader,
        device=device,
        n_batches=args.eval_batches,
    )

    baseline_results = None
    if args.compare_baseline:
        baseline_results = run_baseline_inverse(
            client_model=client_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            cut_layer=args.cut_layer,
            device=device,
            epochs=args.inverse_epochs,
        )

    print(f"\n\n{'=' * 65}")
    print(f"  FINAL RESULTS - PCAT | Cut Layer {args.cut_layer} | CIFAR-10")
    print(f"{'=' * 65}")
    print(f"  SL Model Test Accuracy      : {test_acc * 100:.2f}%")
    print(f"  Stolen Functionality Acc    : {functionality['functionality_acc'] * 100:.2f}%")
    print("")
    print("  Attack              | SSIM    | PSNR")
    print("  ------------------- | ------- | --------")
    print(f"  PCAT                | {pcat_results['ssim']:.4f}  | {pcat_results['psnr']:.2f} dB")
    if baseline_results is not None:
        print(
            f"  Inverse Network     | {baseline_results['ssim']:.4f}  | "
            f"{baseline_results['psnr']:.2f} dB"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "pcat.pt")
    results_path = os.path.join(args.save_dir, "pcat_results.json")
    pcat.save(ckpt_path)

    results = {
        "pcat": pcat_results,
        "baseline": baseline_results,
        "sl_accuracy": test_acc,
        "functionality": functionality,
        "history": sl_history,
        "config": {
            "cut_layer": args.cut_layer,
            "sl_epochs": args.sl_epochs,
            "inverse_epochs": args.inverse_epochs,
            "lambda_moment": args.lambda_moment,
            "eval_batches": args.eval_batches,
        },
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved checkpoint: {ckpt_path}")
    print(f"  Saved results   : {results_path}\n")

    return results


if __name__ == "__main__":
    main()
