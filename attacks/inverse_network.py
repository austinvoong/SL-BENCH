"""
attacks/inverse_network.py
──────────────────────────
Inverse Network Attack — Baseline Semi-Honest Reconstruction Attack

Threat model:
    The server is SEMI-HONEST: it follows the split learning protocol exactly
    (no gradient manipulation, no protocol deviation), but passively attempts to
    reconstruct client inputs from the smashed data it legitimately receives.

Attack mechanism:
    During or after split learning training, the server collects a dataset of
    smashed data tensors (which it sees in every forward pass). It then trains
    a decoder network — the "Inverse Network" — to map smashed data back to the
    original input space, supervised by auxiliary labeled images.

    In the threat model, the server uses auxiliary data from the same domain
    (e.g., a different subset of CIFAR-10 that it runs through the victim's
    client model — approximated here by running through the trained client model
    that the server has observed through gradient updates).

Why this is the right baseline:
    - No protocol deviation → undetectable by gradient monitoring
    - Minimal assumptions → only requires access to auxiliary data
    - Simple architecture → isolates whether smashed data retains pixel-level info
    - FSHA and FORA are both extensions of this core idea; beating this baseline
      is necessary but not sufficient for defending against FORA

Architecture of the Inverse Network (decoder):
    The decoder mirrors the client model's encoder architecture. For SimpleCNN
    with cut_layer=2, smashed data is (64, 8, 8), so the decoder uses transposed
    convolutions to upsample back to (3, 32, 32).

    Smashed shape → decoder design:
        cut_layer=1: (32, 16, 16) → 1 upsample block → (3, 32, 32)
        cut_layer=2: (64,  8,  8) → 2 upsample blocks → (3, 32, 32)
        cut_layer=3: (128, 4,  4) → 3 upsample blocks → (3, 32, 32)

Training:
    Supervised: MSE(decoder(smashed), original_image)
    Optional perceptual regularization to improve visual quality.

Evaluation:
    SSIM and PSNR on a held-out test set, reported before/after defense.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from metrics.reconstruction import compute_ssim, compute_psnr, reconstruction_report


# ─────────────────────────────────────────────────────────────────────────────
# Decoder / Inverse Network architecture
# ─────────────────────────────────────────────────────────────────────────────

class InverseNetwork(nn.Module):
    """
    Decoder network that maps smashed data back to original input images.

    Architecture uses transposed convolutions for learnable upsampling,
    with BatchNorm and ReLU activations matching the encoder style.
    The output is passed through Tanh to bound pixel values.

    Designed to invert SimpleCNN's encoding at cut layers 1, 2, or 3.
    For other architectures, instantiate with explicit in_channels / spatial_size.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        out_channels: int = 3,
        out_size: int = 32,
    ):
        """
        Args:
            in_channels:  Number of channels in smashed data (32, 64, or 128 for SimpleCNN)
            spatial_size: Spatial dimension of smashed data  (16, 8, or 4 for SimpleCNN)
            out_channels: Output channels (3 for RGB)
            out_size:     Target output spatial size (32 for CIFAR-10)
        """
        super().__init__()

        self.in_channels  = in_channels
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.out_size     = out_size

        # Build upsample blocks. Each block doubles the spatial dimension.
        # We need log2(out_size / spatial_size) upsampling stages.
        n_upsample = 0
        s = spatial_size
        while s < out_size:
            s *= 2
            n_upsample += 1

        if s != out_size:
            raise ValueError(
                f"Cannot reach out_size={out_size} from spatial_size={spatial_size} "
                f"by doubling. Use power-of-2 compatible sizes."
            )

        layers = []
        current_channels = in_channels

        # Initial refinement conv (same spatial size)
        layers += [
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
        ]

        # Upsampling blocks
        for i in range(n_upsample):
            out_ch = max(current_channels // 2, out_channels * 4)
            if i == n_upsample - 1:
                out_ch = out_channels * 8  # penultimate block

            layers += [
                nn.ConvTranspose2d(
                    current_channels, out_ch,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            current_channels = out_ch

        # Final projection to output channels
        layers += [
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # bounds output to (-1, 1), matching normalized CIFAR-10
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from smashed data.

        Args:
            smashed: Intermediate activations (B, C_smashed, H_smashed, W_smashed)

        Returns:
            Reconstructed image (B, 3, 32, 32) for CIFAR-10
        """
        return self.decoder(smashed)

    @staticmethod
    def for_cut_layer(cut_layer: int, num_classes: int = 10) -> "InverseNetwork":
        """
        Factory: build an InverseNetwork matching SimpleCNN's smashed data dimensions.

        Args:
            cut_layer: SimpleCNN cut layer (1, 2, or 3)

        Returns:
            Configured InverseNetwork
        """
        configs = {
            1: {"in_channels": 32,  "spatial_size": 16},
            2: {"in_channels": 64,  "spatial_size": 8},
            3: {"in_channels": 128, "spatial_size": 4},
        }
        if cut_layer not in configs:
            raise ValueError(f"cut_layer must be 1, 2, or 3. Got {cut_layer}.")
        return InverseNetwork(**configs[cut_layer])


# ─────────────────────────────────────────────────────────────────────────────
# Attack class
# ─────────────────────────────────────────────────────────────────────────────

class InverseNetworkAttack:
    """
    Semi-honest inverse network reconstruction attack.

    Workflow:
        1. Collect smashed data by running auxiliary images through the
           (trained) client model. The server observes smashed data during
           normal split learning training, so we simulate this by running
           the auxiliary dataset through the frozen client model.
        2. Train the InverseNetwork decoder on (smashed_data, original_image) pairs.
        3. Evaluate reconstruction quality on a held-out test set using SSIM/PSNR.

    The attack is post-hoc (applied after split learning training), which is
    the realistic threat model: the server has been passively collecting smashed
    data throughout training, then launches the attack offline.
    """

    def __init__(
        self,
        client_model: nn.Module,
        cut_layer: int = 2,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        """
        Args:
            client_model: The trained client-side model (server has observed its
                          outputs throughout training via the smashed data stream).
                          In a real attack the server would train a substitute;
                          here we use the actual client model for a strong oracle.
            cut_layer:    Which cut layer was used during split learning (1, 2, or 3)
            lr:           Learning rate for training the inverse network
            device:       Compute device
        """
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Freeze the client model — attacker observes outputs, doesn't modify it
        self.client_model = client_model.to(self.device).eval()
        for p in self.client_model.parameters():
            p.requires_grad_(False)

        self.cut_layer = cut_layer
        self.lr = lr

        # Inverse network (to be trained by the attacker)
        self.inverse_net = InverseNetwork.for_cut_layer(cut_layer).to(self.device)

        self.optimizer = torch.optim.Adam(self.inverse_net.parameters(), lr=lr)

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_ssim":   [],
            "val_psnr":   [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Build the smashed-data dataset
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def build_smashed_dataset(
        self,
        data_loader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> TensorDataset:
        """
        Run auxiliary data through the client model to collect (smashed, original) pairs.

        The server legitimately receives all smashed data during split learning.
        This simulates building that dataset from the auxiliary loader.

        Args:
            data_loader:  Loader yielding (images, labels) — labels are ignored.
                          In practice this is the training loader the server observed.
            max_samples:  If set, stop after collecting this many samples.

        Returns:
            TensorDataset of (smashed_data, original_images)
        """
        self.client_model.eval()

        all_smashed  = []
        all_originals = []
        collected = 0

        print(f"  [Attack] Collecting smashed data from {len(data_loader)} batches...")

        for images, _ in tqdm(data_loader, desc="  Building smashed dataset", leave=False):
            images = images.to(self.device)

            smashed = self.client_model(images)

            all_smashed.append(smashed.cpu())
            all_originals.append(images.cpu())

            collected += images.shape[0]
            if max_samples and collected >= max_samples:
                break

        smashed_tensor   = torch.cat(all_smashed,   dim=0)
        originals_tensor = torch.cat(all_originals, dim=0)

        if max_samples:
            smashed_tensor   = smashed_tensor[:max_samples]
            originals_tensor = originals_tensor[:max_samples]

        print(f"  [Attack] Collected {len(smashed_tensor)} (smashed, original) pairs.")
        print(f"           Smashed data shape per sample: {tuple(smashed_tensor.shape[1:])}")

        return TensorDataset(smashed_tensor, originals_tensor)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Train the inverse network
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: TensorDataset,
        val_loader: Optional[DataLoader],
        epochs: int = 30,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the inverse network to reconstruct inputs from smashed data.

        Loss: MSE(reconstructed, original)
        The MSE loss directly minimizes per-pixel reconstruction error, which
        correlates with PSNR. SSIM is tracked as a validation metric.

        Args:
            train_dataset: TensorDataset from build_smashed_dataset()
            val_loader:    Optional loader of original (images, _) for SSIM/PSNR evaluation
            epochs:        Training epochs
            batch_size:    Mini-batch size for inverse network training
            verbose:       Print per-epoch metrics

        Returns:
            Training history dict
        """
        attack_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # TensorDataset is already in memory
        )

        n_params = sum(p.numel() for p in self.inverse_net.parameters())

        print(f"\n{'─' * 60}")
        print(f"  Inverse Network Attack — Training")
        print(f"{'─' * 60}")
        print(f"  Decoder parameters   : {n_params:,}")
        print(f"  Training pairs       : {len(train_dataset):,}")
        print(f"  Epochs               : {epochs}")
        print(f"{'─' * 60}\n")

        for epoch in range(epochs):
            self.inverse_net.train()
            total_loss = 0.0
            total_samples = 0

            pbar = tqdm(attack_loader, desc=f"  Epoch {epoch+1:3d}/{epochs}", leave=False)
            for smashed, originals in pbar:
                smashed   = smashed.to(self.device)
                originals = originals.to(self.device)

                self.optimizer.zero_grad()
                reconstructed = self.inverse_net(smashed)

                # Primary loss: pixel-level MSE
                loss = F.mse_loss(reconstructed, originals)
                loss.backward()
                self.optimizer.step()

                total_loss    += loss.item() * smashed.shape[0]
                total_samples += smashed.shape[0]

                pbar.set_postfix({"mse": f"{loss.item():.5f}"})

            avg_loss = total_loss / total_samples
            self.history["train_loss"].append(avg_loss)

            # Validation: SSIM + PSNR on original test images
            if val_loader is not None:
                val_ssim, val_psnr = self._evaluate(val_loader)
                self.history["val_ssim"].append(val_ssim)
                self.history["val_psnr"].append(val_psnr)
            else:
                val_ssim = val_psnr = float("nan")

            if verbose:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"MSE Loss: {avg_loss:.5f} | "
                    f"SSIM: {val_ssim:.4f} | "
                    f"PSNR: {val_psnr:.2f} dB"
                )

        return self.history

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Evaluate reconstruction quality
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate SSIM and PSNR on a data loader of original images.

        Runs images through client model → inverse network → metrics.
        """
        self.inverse_net.eval()
        self.client_model.eval()

        all_ssim = []
        all_psnr = []

        for images, _ in data_loader:
            images = images.to(self.device)

            smashed       = self.client_model(images)
            reconstructed = self.inverse_net(smashed)

            all_ssim.append(compute_ssim(reconstructed, images).item())
            all_psnr.append(compute_psnr(reconstructed, images).item())

        return sum(all_ssim) / len(all_ssim), sum(all_psnr) / len(all_psnr)

    @torch.no_grad()
    def evaluate_full(
        self,
        data_loader: DataLoader,
        n_report_batches: int = 4,
    ) -> Dict[str, float]:
        """
        Full evaluation report: SSIM, PSNR, and distance correlation.

        Args:
            data_loader:      Loader of (images, labels) to evaluate on
            n_report_batches: Number of batches to use for distance correlation
                              (dCor is O(N²) so we subsample)

        Returns:
            Dict with ssim, psnr, dcor
        """
        ssim, psnr = self._evaluate(data_loader)

        # Collect a subsample for distance correlation (expensive O(N²) computation)
        all_originals  = []
        all_smashed    = []
        batch_count    = 0

        self.client_model.eval()
        for images, _ in data_loader:
            images = images.to(self.device)
            smashed = self.client_model(images)
            all_originals.append(images)
            all_smashed.append(smashed)
            batch_count += 1
            if batch_count >= n_report_batches:
                break

        originals_sample = torch.cat(all_originals, dim=0)
        smashed_sample   = torch.cat(all_smashed,   dim=0)

        from metrics.reconstruction import distance_correlation
        dcor = distance_correlation(originals_sample, smashed_sample).item()

        reconstruction_report(
            originals=originals_sample,
            reconstructed=self.inverse_net(smashed_sample),
            smashed_data=smashed_sample,
            label="Inverse Network Attack",
        )

        return {"ssim": ssim, "psnr": psnr, "dcor": dcor}

    @torch.no_grad()
    def reconstruct_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of images (convenience method for visualization).

        Args:
            images: Original images (B, C, H, W) — will be encoded then decoded

        Returns:
            Reconstructed images (B, C, H, W)
        """
        self.client_model.eval()
        self.inverse_net.eval()
        images = images.to(self.device)
        smashed = self.client_model(images)
        return self.inverse_net(smashed)

    def save(self, path: str):
        """Save the trained inverse network."""
        torch.save({
            "inverse_net": self.inverse_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cut_layer": self.cut_layer,
            "history": self.history,
        }, path)
        print(f"[Attack] Inverse network saved to {path}")

    def load(self, path: str):
        """Load a previously trained inverse network."""
        ckpt = torch.load(path, map_location=self.device)
        self.inverse_net.load_state_dict(ckpt["inverse_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.history = ckpt["history"]
        print(f"[Attack] Inverse network loaded from {path}")