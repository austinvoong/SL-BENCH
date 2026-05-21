"""
Adversarial Feature Obfuscation (AFO)

AFO is a targeted defense for semi-honest reconstruction attacks such as FORA.
The client inserts a small residual obfuscation module at the cut layer:

    images -> client_model -> smashed -> AFO obfuscator -> server_model

The obfuscator is trained with two competing objectives:
    1. Preserve task utility through the normal split learning classification loss.
    2. Frustrate a local reconstruction attacker trained to invert the defended
       smashed data.

The local reconstructor is not the real attacker. It is a pressure signal used
by the client during training. In evaluation, FORA still observes only the
defended smashed data that the server would legitimately receive.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.inverse_network import InverseNetwork
from metrics.reconstruction import compute_psnr, compute_ssim, distance_correlation


SMASHED_SHAPES = {
    # cut_layer: (channels, spatial)
    1: (32, 16),
    2: (64, 8),
    3: (128, 4),
}


class AFOObfuscator(nn.Module):
    """
    Lightweight residual feature obfuscator.

    The module preserves the smashed-data shape. The residual form keeps the
    defense close to the original representation early in training, which helps
    preserve classification utility.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        strength: float = 0.15,
    ):
        super().__init__()
        hidden = hidden_channels or max(channels // 2, 16)
        self.strength = strength

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        perturbation = torch.tanh(self.net(smashed))
        return smashed + self.strength * perturbation


class AFOTrainer:
    """
    Vanilla split learning trainer augmented with Adversarial Feature Obfuscation.

    This trainer owns three client-side components:
        - client_model: bottom split-learning layers
        - obfuscator: trainable residual AFO module at the cut layer
        - reconstructor: local attacker used only to train the obfuscator

    The server receives only obfuscated smashed data. When a FORA attacker is
    supplied to train(), its substitute model is updated on the same defended
    tensor that the server receives.
    """

    def __init__(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cut_layer: int = 2,
        lr: float = 1e-3,
        lr_obfuscator: float = 5e-4,
        lr_reconstructor: float = 1e-3,
        lambda_reconstruction: float = 0.05,
        lambda_feature: float = 0.02,
        obfuscation_strength: float = 0.15,
        recon_steps: int = 1,
        dcor_subsample: int = 64,
        device: Optional[str] = None,
    ):
        if cut_layer not in SMASHED_SHAPES:
            raise ValueError(f"cut_layer must be one of {sorted(SMASHED_SHAPES)}")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        smashed_channels, _ = SMASHED_SHAPES[cut_layer]

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)
        self.obfuscator = AFOObfuscator(
            channels=smashed_channels,
            strength=obfuscation_strength,
        ).to(self.device)
        self.reconstructor = InverseNetwork.for_cut_layer(cut_layer).to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cut_layer = cut_layer
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_feature = lambda_feature
        self.recon_steps = recon_steps
        self.dcor_subsample = dcor_subsample

        self.client_optimizer = torch.optim.Adam(self.client_model.parameters(), lr=lr)
        self.server_optimizer = torch.optim.Adam(self.server_model.parameters(), lr=lr)
        self.obfuscator_optimizer = torch.optim.Adam(
            self.obfuscator.parameters(), lr=lr_obfuscator
        )
        self.reconstructor_optimizer = torch.optim.Adam(
            self.reconstructor.parameters(), lr=lr_reconstructor
        )

        self.criterion = nn.CrossEntropyLoss()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "test_acc": [],
            "epoch_time": [],
            "afo_recon_loss": [],
            "afo_adv_loss": [],
            "afo_feature_loss": [],
            "dcor_values": [],
            "fora_sub_loss": [],
            "fora_disc_loss": [],
            "fora_mmd_loss": [],
        }

    # ── Public helpers ───────────────────────────────────────────────────────

    def defend_smashed(self, smashed: torch.Tensor) -> torch.Tensor:
        """Apply the learned obfuscator to smashed data."""
        return self.obfuscator(smashed)

    @torch.no_grad()
    def get_defended_smashed(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through client + AFO for evaluation."""
        self.client_model.eval()
        self.obfuscator.eval()
        images = images.to(self.device)
        return self.obfuscator(self.client_model(images))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _set_reconstructor_grad(self, requires_grad: bool) -> None:
        for p in self.reconstructor.parameters():
            p.requires_grad_(requires_grad)

    def _train_reconstructor(
        self,
        defended_smashed: torch.Tensor,
        images: torch.Tensor,
    ) -> float:
        """
        Train the local reconstruction attacker on the defended representation.

        The obfuscator is detached here. This is the attacker's best response
        to the current defense state.
        """
        self.reconstructor.train()
        loss_value = 0.0

        for _ in range(self.recon_steps):
            self.reconstructor_optimizer.zero_grad()
            recon = self.reconstructor(defended_smashed.detach())
            loss = F.mse_loss(recon, images)
            loss.backward()
            self.reconstructor_optimizer.step()
            loss_value = loss.item()

        return loss_value

    def _batch_dcor(self, images: torch.Tensor, defended_smashed: torch.Tensor) -> float:
        n = min(images.size(0), self.dcor_subsample)
        if n < 2:
            return 0.0
        return distance_correlation(images[:n].detach(), defended_smashed[:n].detach()).item()

    # ── Core training ────────────────────────────────────────────────────────

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        fora_attack=None,
        collect_snapshot: bool = False,
    ) -> Dict[str, float]:
        """
        Run one AFO split-learning training step.

        If fora_attack is provided, it is updated on defended smashed data before
        the model update, matching the semi-honest server observation stream.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.client_optimizer.zero_grad()
        self.obfuscator_optimizer.zero_grad()

        # Client-side forward: raw smashed data then defended smashed data.
        smashed = self.client_model(images)
        defended = self.obfuscator(smashed)

        # Train the local reconstructor to invert the current defended signal.
        recon_loss = self._train_reconstructor(defended, images)

        # FORA observes exactly what the server receives.
        fora_metrics = {"sub_loss": float("nan"), "disc_loss": float("nan"), "mmd_loss": float("nan")}
        if fora_attack is not None:
            fora_metrics = fora_attack.update_substitute(defended.detach())
            if collect_snapshot:
                fora_attack.add_to_snapshot(defended.detach())

        # Server-side task update through the split boundary.
        defended_server = defended.detach().requires_grad_(True)
        self.server_optimizer.zero_grad()
        outputs = self.server_model(defended_server)
        task_loss = self.criterion(outputs, labels)
        task_loss.backward()
        grad_to_client = defended_server.grad.clone()
        self.server_optimizer.step()

        # Backpropagate task gradient to client and obfuscator.
        defended.backward(grad_to_client, retain_graph=True)

        # Train the obfuscator to increase reconstruction error while staying
        # near the original feature manifold. The client model is detached for
        # this adversarial part, so the obfuscator absorbs the privacy pressure.
        self._set_reconstructor_grad(False)
        defended_adv = self.obfuscator(smashed.detach())
        recon_adv = self.reconstructor(defended_adv)
        adv_recon_loss = F.mse_loss(recon_adv, images)
        feature_loss = F.mse_loss(defended_adv, smashed.detach())
        afo_loss = (
            self.lambda_feature * feature_loss
            - self.lambda_reconstruction * adv_recon_loss
        )
        afo_loss.backward()
        self._set_reconstructor_grad(True)

        self.client_optimizer.step()
        self.obfuscator_optimizer.step()

        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).sum().item() / labels.size(0)
        dcor = self._batch_dcor(images, defended)

        return {
            "task_loss": task_loss.item(),
            "train_acc": acc,
            "recon_loss": recon_loss,
            "adv_loss": adv_recon_loss.item(),
            "feature_loss": feature_loss.item(),
            "dcor": dcor,
            "fora_sub_loss": fora_metrics["sub_loss"],
            "fora_disc_loss": fora_metrics["disc_loss"],
            "fora_mmd_loss": fora_metrics["mmd_loss"],
        }

    def train(
        self,
        epochs: int,
        fora_attack=None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train AFO split learning.

        Args:
            epochs: Number of SL/AFO epochs.
            fora_attack: Optional FORAAttack instance. If provided, Phase 1 is
                trained in parallel on defended smashed data.
            verbose: Print epoch summaries.
        """
        print("\nStarting AFO Split Learning Training")
        print("=" * 50)
        print(f"Client parameters      : {sum(p.numel() for p in self.client_model.parameters()):,}")
        print(f"Server parameters      : {sum(p.numel() for p in self.server_model.parameters()):,}")
        print(f"AFO parameters         : {sum(p.numel() for p in self.obfuscator.parameters()):,}")
        print(f"Local recon parameters : {sum(p.numel() for p in self.reconstructor.parameters()):,}")
        print("=" * 50)

        last_epoch = epochs - 1

        for epoch in range(epochs):
            start = time.time()
            self.client_model.train()
            self.server_model.train()
            self.obfuscator.train()

            totals = {
                "task_loss": 0.0,
                "correct": 0.0,
                "samples": 0,
                "recon_loss": [],
                "adv_loss": [],
                "feature_loss": [],
                "dcor": [],
                "fora_sub_loss": [],
                "fora_disc_loss": [],
                "fora_mmd_loss": [],
            }

            pbar = tqdm(
                self.train_loader,
                desc=f"  [afo] Epoch {epoch+1:3d}/{epochs}",
                leave=False,
            )

            for images, labels in pbar:
                metrics = self.train_step(
                    images,
                    labels,
                    fora_attack=fora_attack,
                    collect_snapshot=(epoch == last_epoch),
                )
                bs = labels.size(0)
                totals["task_loss"] += metrics["task_loss"] * bs
                totals["correct"] += metrics["train_acc"] * bs
                totals["samples"] += bs

                for key in (
                    "recon_loss",
                    "adv_loss",
                    "feature_loss",
                    "dcor",
                    "fora_sub_loss",
                    "fora_disc_loss",
                    "fora_mmd_loss",
                ):
                    value = metrics[key]
                    if value == value:  # filters NaN without importing math
                        totals[key].append(value)

                pbar.set_postfix({
                    "loss": f"{metrics['task_loss']:.4f}",
                    "adv": f"{metrics['adv_loss']:.4f}",
                    "dcor": f"{metrics['dcor']:.3f}",
                })

            avg_loss = totals["task_loss"] / max(totals["samples"], 1)
            avg_acc = totals["correct"] / max(totals["samples"], 1)
            test_acc = self.evaluate(self.test_loader)
            elapsed = time.time() - start

            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(avg_acc)
            self.history["test_acc"].append(test_acc)
            self.history["epoch_time"].append(elapsed)
            self.history["afo_recon_loss"].append(_avg(totals["recon_loss"]))
            self.history["afo_adv_loss"].append(_avg(totals["adv_loss"]))
            self.history["afo_feature_loss"].append(_avg(totals["feature_loss"]))
            self.history["dcor_values"].append(_avg(totals["dcor"]))
            self.history["fora_sub_loss"].append(_avg(totals["fora_sub_loss"]))
            self.history["fora_disc_loss"].append(_avg(totals["fora_disc_loss"]))
            self.history["fora_mmd_loss"].append(_avg(totals["fora_mmd_loss"]))

            if verbose:
                print(
                    f"  [afo] Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train Acc: {avg_acc*100:5.2f}% | "
                    f"Test Acc: {test_acc*100:5.2f}% | "
                    f"Recon: {self.history['afo_recon_loss'][-1]:.4f} | "
                    f"Adv: {self.history['afo_adv_loss'][-1]:.4f} | "
                    f"dCor: {self.history['dcor_values'][-1]:.4f}"
                )

        return self.history

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> float:
        """Return classification accuracy with AFO active."""
        loader = loader or self.test_loader
        self.client_model.eval()
        self.server_model.eval()
        self.obfuscator.eval()

        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            smashed = self.client_model(images)
            defended = self.obfuscator(smashed)
            outputs = self.server_model(defended)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)

        return correct / max(total, 1)

    @torch.no_grad()
    def reconstruction_probe(
        self,
        loader: Optional[DataLoader] = None,
        n_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the local reconstructor. This is a diagnostic for AFO training,
        not the final FORA attack result.
        """
        loader = loader or self.test_loader
        self.client_model.eval()
        self.obfuscator.eval()
        self.reconstructor.eval()

        ssims = []
        psnrs = []

        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(self.device)
            defended = self.obfuscator(self.client_model(images))
            recon = self.reconstructor(defended).clamp(-1, 1)
            ssims.append(compute_ssim(recon, images).item())
            psnrs.append(compute_psnr(recon, images).item())

        return {"ssim": _avg(ssims), "psnr": _avg(psnrs)}

    def save_checkpoint(self, path: str) -> None:
        """Save AFO defense state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "client_model": self.client_model.state_dict(),
            "server_model": self.server_model.state_dict(),
            "obfuscator": self.obfuscator.state_dict(),
            "reconstructor": self.reconstructor.state_dict(),
            "history": self.history,
            "cut_layer": self.cut_layer,
            "lambda_reconstruction": self.lambda_reconstruction,
            "lambda_feature": self.lambda_feature,
        }, path)
        print(f"[AFO] Saved to {path}")


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
