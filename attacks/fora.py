"""
attacks/fora.py
───────────────
Feature-Oriented Reconstruction Attack (FORA) — Semi-Honest Reconstruction Attack

Reference:
    Xu et al., "A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack
    against Split Learning" (2024)
    https://arxiv.org/abs/2405.04115
    Code: https://github.com/X1aoyangXu/FORA

Threat model:
    SEMI-HONEST server. No protocol deviation — the server follows the split
    learning protocol exactly. It only uses smashed data it legitimately
    receives during training, plus publicly available auxiliary images.

Core insight (from the paper §3.2):
    Even models trained under identical settings develop distinct "representation
    preferences" — the client model's unique pattern of attention and feature
    extraction. This preference is revealed in the smashed data stream. FORA
    trains a substitute client to mimic this preference via domain adaptation,
    then trains an inverse network to map substitute smashed data → images.

Three phases:

Phase 1 — Substitute Model Construction (runs IN PARALLEL with SL training):
    - Server initializes substitute client F̂c (VGG-like blocks)
    - At each training step server receives Zpriv = Fc(Xpriv) [legitimate]
    - Server feeds Xaux through F̂c to get Zaux = F̂c(Xaux)
    - F̂c and Discriminator D are co-trained via:

        Discriminator loss (eq. 2):
          L_D = log(1 − D(Zpriv)) + log(D(Zaux))
          → D tries to classify Zpriv as real (D→1) and Zaux as fake (D→0)

        Substitute client loss (eqs. 1, 3):
          L_DISC    = log(1 − D(Zaux))        [adversarial: fool D, push Zaux→1]
          L_MK-MMD  = ||φ(Zaux) − φ(Zpriv)||_H  [distribution alignment in RKHS]
          L_total   = L_DISC + L_MK-MMD

Phase 2 — Attack Model Training (AFTER SL training):
    - Train inverse network f_c^{-1} on auxiliary data supervised by F̂c:
        L_{f_c^{-1}} = || f_c^{-1}(F̂c(Xaux)) − Xaux ||_2^2

Phase 3 — Private Data Reconstruction:
    - Collect snapshot of smashed data from FINAL training iteration:
        Zsnap = Fc(Xpriv)
    - Reconstruct:
        X*_priv = f_c^{-1}(Zsnap)

Why FORA beats existing defenses:
    - NoPeekNN: reduces statistical correlation of smashed data with raw input,
      but FORA learns feature-space BEHAVIOR, not input-output mappings.
      Statistical independence does not prevent behavioral mimicry.
    - DP: requires ε ≤ 0.1 (10-15% accuracy drop) to stop FORA. At ε=5,
      SSIM remains ~75% of undefended. This motivates our novel defenses.

Design notes for SL-BENCH:
    - Substitute client architecture is auto-constructed from smashed data shape
      to avoid hardcoding to a single cut layer.
    - MK-MMD uses 5 RBF kernels with bandwidths spanning multiple scales,
      following the standard DAN/DANN practice (Long et al. 2015).
    - The discriminator is a simple convolutional stack that processes smashed
      data in its native spatial form (no premature flattening).
    - The inverse network reuses the same design as InverseNetwork in
      inverse_network.py for consistency.
"""

import time
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from metrics.reconstruction import compute_ssim, compute_psnr, reconstruction_report


# ─────────────────────────────────────────────────────────────────────────────
# Substitute Client (F̂c)
# ─────────────────────────────────────────────────────────────────────────────

class SubstituteClient(nn.Module):
    """
    Server-side substitute model that mimics the victim client's feature
    extraction behavior.

    Architecture: VGG-like blocks (Conv + BN + ReLU + MaxPool) that downsample
    the auxiliary input to match the smashed data's spatial resolution and
    channel count. The server does NOT know the victim's architecture (per the
    threat model); these VGG blocks are a reasonable domain-agnostic choice
    that the paper validates empirically (Table 9, also §4.5 architecture ablation).

    Output shape is guaranteed to match smashed data shape exactly so that the
    MK-MMD and discriminator losses are well-defined.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        out_spatial: int = 8,
        in_spatial: int = 32,
    ):
        """
        Args:
            in_channels:  Input image channels (3 for RGB)
            out_channels: Target smashed data channel count
            out_spatial:  Target smashed data spatial size (H = W assumed)
            in_spatial:   Input image spatial size
        """
        super().__init__()

        self.out_channels = out_channels
        self.out_spatial  = out_spatial

        # Number of MaxPool(2) downsampling stages needed
        n_downsample = 0
        s = in_spatial
        while s > out_spatial:
            s //= 2
            n_downsample += 1

        if s != out_spatial:
            raise ValueError(
                f"Cannot reach out_spatial={out_spatial} from in_spatial={in_spatial} "
                f"by halving. Use power-of-2 compatible sizes."
            )

        # Build VGG-like blocks
        layers: List[nn.Module] = []
        current_ch = in_channels

        for i in range(n_downsample):
            # Last block outputs exactly out_channels
            out_ch = out_channels if i == n_downsample - 1 else max(out_channels // (2 ** (n_downsample - 1 - i)), 32)
            layers += [
                nn.Conv2d(current_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]
            current_ch = out_ch

        # Optional refinement conv to ensure exact channel count
        if current_ch != out_channels:
            layers += [
                nn.Conv2d(current_ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    @staticmethod
    def for_smashed_shape(
        smashed_channels: int,
        smashed_spatial: int,
        img_channels: int = 3,
        img_spatial: int = 32,
    ) -> "SubstituteClient":
        """Factory: build substitute client matching a given smashed data shape."""
        return SubstituteClient(
            in_channels=img_channels,
            out_channels=smashed_channels,
            out_spatial=smashed_spatial,
            in_spatial=img_spatial,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator (D)
# ─────────────────────────────────────────────────────────────────────────────

class SmashDiscriminator(nn.Module):
    """
    Discriminator that distinguishes victim smashed data Zpriv from substitute
    smashed data Zaux.

    Architecture: Convolutional layers with stride-2 (spatial compression) →
    Global average pooling → Linear → sigmoid probability.

    We keep this architecture simple but effective. The paper's discriminator
    (Table 9) uses residual blocks; our version uses plain convolutions to avoid
    training instability. The GAN training for FORA uses standard binary
    cross-entropy, not Wasserstein loss (unlike FSHA which used WGAN-GP).

    Input shape: (B, C, H, W) — the smashed data spatial tensor
    Output shape: (B, 1) — probability that input is Zpriv (real)
    """

    def __init__(self, in_channels: int, in_spatial: int):
        """
        Args:
            in_channels: Channel count of smashed data
            in_spatial:  Spatial size of smashed data (H = W)
        """
        super().__init__()

        layers: List[nn.Module] = []
        ch = in_channels

        # Downsample until spatial size reaches 2 or we've done 3 conv blocks
        n_blocks = min(3, int(torch.log2(torch.tensor(float(in_spatial))).item()) - 1)

        for i in range(n_blocks):
            out_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        # Global average pooling collapses spatial dims regardless of resolution
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Smashed data (B, C, H, W)
        Returns:
            Probability tensor (B, 1) — sigmoid NOT applied (use BCEWithLogitsLoss)
        """
        h = self.conv_layers(x)
        h = self.gap(h)
        h = h.view(h.size(0), -1)
        return self.linear(h)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Kernel Maximum Mean Discrepancy (MK-MMD)
# ─────────────────────────────────────────────────────────────────────────────

def mk_mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    bandwidth_list: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Multi-Kernel Maximum Mean Discrepancy (MK-MMD).

    Measures the distance between two distributions P (source) and Q (target)
    in a Reproducing Kernel Hilbert Space (RKHS) using a mixture of RBF kernels.

    MK-MMD²(P,Q) = E_{x,x'~P}[k(x,x')] + E_{y,y'~Q}[k(y,y')] - 2·E_{x~P,y~Q}[k(x,y)]

    where k = Σ_j β_j k_j is a convex combination of RBF kernels:
        k_j(x, y) = exp(−||x − y||² / (2σ_j²))

    Following Long et al. (2015) DAN paper convention, we use 5 kernels with
    bandwidth scale factors [2^{-2}, 2^{-1}, 1, 2^1, 2^2] multiplied by the
    median pairwise distance (bandwidth heuristic).

    Reference: Long et al. "Learning Transferable Features with Deep Adaptation
    Networks" (ICML 2015) — equations 2-3.

    Args:
        source: Feature tensor from substitute client  (B, C, H, W) or (B, D)
        target: Feature tensor from victim client      (B, C, H, W) or (B, D)
        bandwidth_list: Optional explicit bandwidth values. If None, computed
                        from the median pairwise distance heuristic.

    Returns:
        Scalar MK-MMD² loss value
    """
    # Flatten spatial dimensions: (B, C, H, W) → (B, C*H*W)
    if source.dim() > 2:
        source = source.reshape(source.size(0), -1)
    if target.dim() > 2:
        target = target.reshape(target.size(0), -1)

    # Cast to float32 for numerical stability
    source = source.float()
    target = target.float()

    n_s = source.size(0)
    n_t = target.size(0)

    # Concatenate for pairwise distance computation: (n_s + n_t, D)
    total = torch.cat([source, target], dim=0)

    # Pairwise squared Euclidean distances: (n_s+n_t, n_s+n_t)
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    sq_norms = (total ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    dists_sq = sq_norms + sq_norms.T - 2.0 * torch.mm(total, total.T)
    dists_sq = dists_sq.clamp(min=0.0)  # numerical safety

    if bandwidth_list is None:
        # Median bandwidth heuristic (standard for MMD)
        with torch.no_grad():
            median_sq = dists_sq.detach().median().clamp(min=1e-6)
        # 5 kernels spanning 2 decades around the median bandwidth
        bandwidth_list = [median_sq * (2.0 ** k) for k in [-2, -1, 0, 1, 2]]

    # Compute MK-MMD using the multi-kernel sum
    # Partition the pairwise distance matrix into SS, ST, TT blocks
    # source indices: 0..n_s-1   target indices: n_s..n_s+n_t-1
    d_ss = dists_sq[:n_s, :n_s]           # (n_s, n_s)
    d_tt = dists_sq[n_s:, n_s:]           # (n_t, n_t)
    d_st = dists_sq[:n_s, n_s:]           # (n_s, n_t)

    mmd_sq = torch.zeros(1, device=source.device, dtype=source.dtype)

    for bw in bandwidth_list:
        k_ss = torch.exp(-d_ss / (2.0 * bw))
        k_tt = torch.exp(-d_tt / (2.0 * bw))
        k_st = torch.exp(-d_st / (2.0 * bw))

        # Unbiased MMD estimator: exclude diagonal terms
        mask_s = 1.0 - torch.eye(n_s, device=source.device)
        mask_t = 1.0 - torch.eye(n_t, device=target.device)

        e_ss = (k_ss * mask_s).sum() / max(n_s * (n_s - 1), 1)
        e_tt = (k_tt * mask_t).sum() / max(n_t * (n_t - 1), 1)
        e_st = k_st.mean()

        mmd_sq = mmd_sq + e_ss + e_tt - 2.0 * e_st

    return mmd_sq.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Inverse Network (f_c^{-1}) — reused architecture from inverse_network.py
# ─────────────────────────────────────────────────────────────────────────────

class FORAInverseNetwork(nn.Module):
    """
    Inverse network f_c^{-1} that maps smashed data back to image space.

    Trained on (F̂c(Xaux), Xaux) pairs in Phase 2 using MSE loss.
    At reconstruction time, applied to real victim smashed data Zsnap.

    Architecture closely follows Table 9 of the FORA paper:
        ConvTranspose2d (stride=2, upsample) blocks → Conv2d refinement → Tanh

    For SimpleCNN cut_layer=2: (64, 8, 8) → (3, 32, 32)
    For SimpleCNN cut_layer=1: (32, 16, 16) → (3, 32, 32)
    For SimpleCNN cut_layer=3: (128, 4, 4) → (3, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        in_spatial: int,
        out_channels: int = 3,
        out_spatial: int = 32,
        mid_channels: int = 256,
    ):
        super().__init__()

        # Number of upsampling stages needed
        n_up = 0
        s = in_spatial
        while s < out_spatial:
            s *= 2
            n_up += 1

        if s != out_spatial:
            raise ValueError(
                f"Cannot reach out_spatial={out_spatial} from in_spatial={in_spatial} "
                f"by doubling."
            )

        layers: List[nn.Module] = []
        ch = in_channels

        # Initial projection to mid_channels if needed
        if ch != mid_channels:
            layers += [
                nn.Conv2d(ch, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ]
            ch = mid_channels

        # Transposed convolution upsampling blocks
        for i in range(n_up):
            out_ch = mid_channels if i < n_up - 1 else mid_channels // 2
            out_ch = max(out_ch, out_channels * 4)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch

        # Refinement conv + final projection
        layers += [
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def for_cut_layer(cut_layer: int) -> "FORAInverseNetwork":
        """Factory for SimpleCNN cut layers 1, 2, 3."""
        configs = {
            1: (32, 16),
            2: (64,  8),
            3: (128, 4),
        }
        if cut_layer not in configs:
            raise ValueError(f"cut_layer must be 1, 2, or 3. Got {cut_layer}.")
        in_ch, in_sp = configs[cut_layer]
        return FORAInverseNetwork(in_channels=in_ch, in_spatial=in_sp)


# ─────────────────────────────────────────────────────────────────────────────
# FORA Attack — Main class
# ─────────────────────────────────────────────────────────────────────────────

class FORAAttack:
    """
    Feature-Oriented Reconstruction Attack (FORA).

    Usage pattern (integrated with a split learning training loop):

        # Setup
        fora = FORAAttack(
            smashed_channels=64, smashed_spatial=8,
            aux_loader=aux_loader, cut_layer=2
        )

        # During SL training — call once per training step:
        for images, labels in train_loader:
            smashed = client_model(images)  # grab a copy before detach
            ... normal SL training step ...
            fora.update_substitute(smashed.detach())

        # After SL training:
        fora.train_inverse_network(epochs=30)

        # Collect snapshot and reconstruct:
        fora.add_to_snapshot(smashed_data)
        reconstructed = fora.reconstruct_from_snapshot()

        # Evaluate:
        fora.evaluate(original_images, reconstructed)
    """

    def __init__(
        self,
        smashed_channels: int,
        smashed_spatial: int,
        aux_loader: DataLoader,
        cut_layer: int = 2,
        img_channels: int = 3,
        img_spatial: int = 32,
        lr_substitute: float = 1e-4,
        lr_discriminator: float = 1e-4,
        lr_inverse: float = 1e-3,
        lambda_mmd: float = 1.0,
        device: Optional[str] = None,
    ):
        """
        Args:
            smashed_channels:  Channel count of smashed data (from cut layer)
            smashed_spatial:   Spatial size of smashed data (H=W)
            aux_loader:        DataLoader of auxiliary images (publicly available,
                               same domain but different distribution from private data)
            cut_layer:         Cut layer index (for InverseNetwork factory)
            img_channels:      Input image channels (3 for RGB)
            img_spatial:       Input image spatial size (32 for CIFAR-10)
            lr_substitute:     Learning rate for substitute client F̂c
            lr_discriminator:  Learning rate for discriminator D
            lr_inverse:        Learning rate for inverse network f_c^{-1}
            lambda_mmd:        Weight of MK-MMD term relative to L_DISC
            device:            Compute device (auto-detected if None)
        """
        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"[FORA] Using device: {self.device}")

        self.smashed_channels = smashed_channels
        self.smashed_spatial  = smashed_spatial
        self.cut_layer        = cut_layer
        self.lambda_mmd       = lambda_mmd

        # ── Auxiliary data ────────────────────────────────────────────────────
        self.aux_loader       = aux_loader
        self._aux_iter: Optional[Iterator] = None

        # ── Networks ──────────────────────────────────────────────────────────
        # Phase 1: substitute client + discriminator
        self.substitute = SubstituteClient.for_smashed_shape(
            smashed_channels=smashed_channels,
            smashed_spatial=smashed_spatial,
            img_channels=img_channels,
            img_spatial=img_spatial,
        ).to(self.device)

        self.discriminator = SmashDiscriminator(
            in_channels=smashed_channels,
            in_spatial=smashed_spatial,
        ).to(self.device)

        # Phase 2: inverse network
        self.inverse_net = FORAInverseNetwork.for_cut_layer(cut_layer).to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        self.opt_substitute     = torch.optim.Adam(
            self.substitute.parameters(), lr=lr_substitute, betas=(0.5, 0.999)
        )
        self.opt_discriminator  = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999)
        )
        self.opt_inverse        = torch.optim.Adam(
            self.inverse_net.parameters(), lr=lr_inverse
        )

        # BCE for discriminator / adversarial loss (no sigmoid in discriminator,
        # use BCEWithLogitsLoss for numerical stability)
        self.bce = nn.BCEWithLogitsLoss()

        # ── Snapshot buffer for private smashed data ──────────────────────────
        # Stores smashed data snapshots from the FINAL training iteration
        self._snapshot_list: List[torch.Tensor] = []

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "disc_loss":   [],
            "sub_loss":    [],
            "mmd_loss":    [],
            "disc_loss_adv": [],
            "inverse_loss": [],
            "val_ssim":    [],
            "val_psnr":    [],
        }

        # Step counters
        self._sub_steps    = 0
        self._inverse_steps = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Auxiliary data iterator (cycles indefinitely)
    # ──────────────────────────────────────────────────────────────────────────

    def _next_aux_batch(self) -> torch.Tensor:
        """
        Return the next batch of auxiliary images, cycling the loader.
        Auxiliary labels are ignored (we only need the images).
        """
        if self._aux_iter is None:
            self._aux_iter = iter(self.aux_loader)

        try:
            batch = next(self._aux_iter)
        except StopIteration:
            self._aux_iter = iter(self.aux_loader)
            batch = next(self._aux_iter)

        # Handle (images, labels) or plain images
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        return images.to(self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Update substitute client (called during SL training)
    # ──────────────────────────────────────────────────────────────────────────

    def update_substitute(self, smashed_priv: torch.Tensor) -> Dict[str, float]:
        """
        One substitute model update step (Phase 1).

        Called at each SL training iteration with the smashed data the server
        legitimately receives from the victim client.

        Steps:
          1. Sample auxiliary batch Xaux
          2. Compute Zaux = F̂c(Xaux)
          3. Update Discriminator D on (Zpriv=real, Zaux=fake)
          4. Update substitute F̂c via adversarial + MK-MMD loss

        Args:
            smashed_priv: Smashed data from victim client (B, C, H, W) — detached,
                          as the server receives it without gradients.

        Returns:
            Dict with step losses for logging
        """
        smashed_priv = smashed_priv.to(self.device).detach()
        batch_size   = smashed_priv.size(0)
        xaux         = self._next_aux_batch()

        # Match batch sizes (auxiliary might differ)
        if xaux.size(0) > batch_size:
            xaux = xaux[:batch_size]
        elif xaux.size(0) < batch_size:
            # Tile auxiliary batch up to batch_size
            repeats = (batch_size // xaux.size(0)) + 1
            xaux    = xaux.repeat(repeats, 1, 1, 1)[:batch_size]

        # ── Step A: Train Discriminator D ─────────────────────────────────────
        # D(Zpriv) → 1 (real), D(Zaux) → 0 (fake)
        # L_D = log(1 − D(Zpriv)) + log(D(Zaux))
        # Using BCEWithLogitsLoss: targets real=1, fake=0
        self.opt_discriminator.zero_grad()

        with torch.no_grad():
            zaux = self.substitute(xaux)  # don't want grads through F̂c here

        real_logits = self.discriminator(smashed_priv)
        fake_logits = self.discriminator(zaux.detach())

        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        loss_d_real = self.bce(real_logits, real_labels)
        loss_d_fake = self.bce(fake_logits, fake_labels)
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        loss_d.backward()
        self.opt_discriminator.step()

        # ── Step B: Train Substitute Client F̂c ──────────────────────────────
        # L_DISC    = log(1 − D(Zaux)) → fool D (want D(Zaux) → 1)
        # L_MK-MMD  = MMD(Zaux, Zpriv)
        self.opt_substitute.zero_grad()

        zaux = self.substitute(xaux)  # recompute with grad
        fake_logits_for_gen = self.discriminator(zaux)

        # Adversarial loss: F̂c wants discriminator to classify its output as real
        loss_adv = self.bce(fake_logits_for_gen, torch.ones_like(fake_logits_for_gen))

        # MK-MMD loss: align Zaux distribution with Zpriv distribution
        loss_mmd = mk_mmd_loss(zaux, smashed_priv.detach())

        loss_sub = loss_adv + self.lambda_mmd * loss_mmd

        loss_sub.backward()
        self.opt_substitute.step()

        self._sub_steps += 1

        # Log
        step_metrics = {
            "disc_loss":    loss_d.item(),
            "sub_loss":     loss_sub.item(),
            "disc_loss_adv": loss_adv.item(),
            "mmd_loss":     loss_mmd.item(),
        }
        self.history["disc_loss"].append(loss_d.item())
        self.history["sub_loss"].append(loss_sub.item())
        self.history["disc_loss_adv"].append(loss_adv.item())
        self.history["mmd_loss"].append(loss_mmd.item())

        return step_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: Train inverse network (called AFTER SL training)
    # ──────────────────────────────────────────────────────────────────────────

    def train_inverse_network(
        self,
        epochs: int = 30,
        verbose: bool = True,
    ) -> None:
        """
        Train the inverse network f_c^{-1} on auxiliary data (Phase 2).

        Loss: MSE(f_c^{-1}(F̂c(Xaux)), Xaux)

        The substitute client F̂c is FROZEN during this phase. We are training
        the inverse network to invert F̂c's encoding of auxiliary images. Since
        F̂c has learned to mimic Fc's representation preference, the resulting
        inverse network can also invert Fc(Xpriv) ≈ F̂c(Xpriv).

        Args:
            epochs:  Number of training epochs
            verbose: Print per-epoch loss
        """
        # Freeze substitute client during inverse network training
        for p in self.substitute.parameters():
            p.requires_grad_(False)

        n_params = sum(p.numel() for p in self.inverse_net.parameters())

        print(f"\n{'─' * 60}")
        print(f"  FORA Phase 2 — Inverse Network Training")
        print(f"{'─' * 60}")
        print(f"  Inverse network parameters : {n_params:,}")
        print(f"  Epochs                     : {epochs}")
        print(f"{'─' * 60}\n")

        self.substitute.eval()
        self.inverse_net.train()

        for epoch in range(epochs):
            total_loss    = 0.0
            total_samples = 0

            pbar = tqdm(
                self.aux_loader,
                desc=f"  [Phase 2] Epoch {epoch+1:3d}/{epochs}",
                leave=False,
            )

            for batch in pbar:
                xaux = batch[0] if isinstance(batch, (list, tuple)) else batch
                xaux = xaux.to(self.device)

                self.opt_inverse.zero_grad()

                with torch.no_grad():
                    zaux = self.substitute(xaux)

                x_recon = self.inverse_net(zaux)
                loss    = F.mse_loss(x_recon, xaux)

                loss.backward()
                self.opt_inverse.step()

                total_loss    += loss.item() * xaux.size(0)
                total_samples += xaux.size(0)

                pbar.set_postfix({"mse": f"{loss.item():.5f}"})

            avg_loss = total_loss / max(total_samples, 1)
            self.history["inverse_loss"].append(avg_loss)
            self._inverse_steps += 1

            if verbose:
                print(
                    f"  [Phase 2] Epoch {epoch+1:3d}/{epochs} | "
                    f"MSE Loss: {avg_loss:.5f}"
                )

        # Unfreeze substitute client
        for p in self.substitute.parameters():
            p.requires_grad_(True)

        print(f"\n  [Phase 2] Training complete. "
              f"Final MSE: {self.history['inverse_loss'][-1]:.5f}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: Snapshot collection and reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    def add_to_snapshot(self, smashed_priv: torch.Tensor) -> None:
        """
        Add a batch of smashed data to the reconstruction snapshot buffer.

        Call this during the FINAL training iteration to collect Zsnap = Fc(Xpriv).
        The paper uses the final iteration's smashed data for reconstruction,
        as the substitute model has converged by then.

        Args:
            smashed_priv: Batch of victim smashed data (B, C, H, W), detached.
        """
        self._snapshot_list.append(smashed_priv.detach().cpu())

    def clear_snapshot(self) -> None:
        """Clear the snapshot buffer."""
        self._snapshot_list.clear()

    @torch.no_grad()
    def reconstruct_from_snapshot(self) -> torch.Tensor:
        """
        Reconstruct private images from collected smashed data snapshots (Phase 3).

        X*_priv = f_c^{-1}(Zsnap)

        Returns:
            Reconstructed images (N, C, H, W) on CPU
        """
        if not self._snapshot_list:
            raise RuntimeError(
                "Snapshot buffer is empty. Call add_to_snapshot() during the "
                "final training iteration before reconstructing."
            )

        self.inverse_net.eval()

        all_recon = []
        zsnap = torch.cat(self._snapshot_list, dim=0)

        batch_size = 128
        for i in range(0, len(zsnap), batch_size):
            z_batch = zsnap[i : i + batch_size].to(self.device)
            recon   = self.inverse_net(z_batch)
            all_recon.append(recon.cpu())

        return torch.cat(all_recon, dim=0)

    @torch.no_grad()
    def reconstruct_batch(
        self,
        smashed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct a batch directly from smashed data (no snapshot needed).

        Args:
            smashed: Smashed data tensor (B, C, H, W)

        Returns:
            Reconstructed images (B, 3, H_out, W_out)
        """
        self.inverse_net.eval()
        return self.inverse_net(smashed.to(self.device))

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        original_images: torch.Tensor,
        reconstructed: torch.Tensor,
        label: str = "FORA",
    ) -> Dict[str, float]:
        """
        Compute SSIM and PSNR between originals and reconstructions.

        Args:
            original_images: Ground-truth images (N, 3, H, W)
            reconstructed:   Reconstructed images (N, 3, H, W)
            label:           Label for the report

        Returns:
            Dict with ssim and psnr
        """
        original_images = original_images.to(self.device)
        reconstructed   = reconstructed.to(self.device)

        ssim = compute_ssim(reconstructed, original_images).item()
        psnr = compute_psnr(reconstructed, original_images).item()

        print(f"\n{'─' * 50}")
        print(f"  {label} Reconstruction Metrics")
        print(f"{'─' * 50}")
        print(f"  SSIM : {ssim:.4f}  (higher=better, 1.0=perfect)")
        print(f"  PSNR : {psnr:.2f} dB  (higher=better)")
        print(f"{'─' * 50}\n")

        return {"ssim": ssim, "psnr": psnr}

    @torch.no_grad()
    def evaluate_on_loader(
        self,
        data_loader: DataLoader,
        client_model: nn.Module,
    ) -> Dict[str, float]:
        """
        End-to-end evaluation: for each batch, encode via client_model,
        reconstruct via inverse_net, and compute SSIM/PSNR.

        Args:
            data_loader:  Loader yielding (images, labels)
            client_model: Trained victim client model

        Returns:
            Dict with avg ssim and psnr
        """
        client_model.eval()
        self.inverse_net.eval()

        all_ssim = []
        all_psnr = []

        for images, _ in data_loader:
            images  = images.to(self.device)
            smashed = client_model(images)
            recon   = self.inverse_net(smashed)

            all_ssim.append(compute_ssim(recon, images).item())
            all_psnr.append(compute_psnr(recon, images).item())

        ssim = sum(all_ssim) / len(all_ssim)
        psnr = sum(all_psnr) / len(all_psnr)

        return {"ssim": ssim, "psnr": psnr}

    # ──────────────────────────────────────────────────────────────────────────
    # Substitute client quality metrics
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def measure_substitute_quality(
        self,
        client_model: nn.Module,
        data_loader: DataLoader,
        n_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Measure how well the substitute client mimics the victim client.

        Computes MSE and cosine similarity between Zpriv and Zaux for
        matched inputs run through both models. Lower MSE and higher cosine
        similarity indicate the substitute has converged.

        Args:
            client_model: Victim client model
            data_loader:  Data loader (using victim's data for fair comparison)
            n_batches:    Number of batches to evaluate on

        Returns:
            Dict with mse and cosine_similarity
        """
        client_model.eval()
        self.substitute.eval()

        all_mse  = []
        all_cos  = []
        batch_n  = 0

        for images, _ in data_loader:
            images = images.to(self.device)

            zpriv = client_model(images)
            zaux  = self.substitute(images)  # same images through substitute

            # Flatten for distance computation
            zp_flat = zpriv.reshape(zpriv.size(0), -1)
            za_flat = zaux.reshape(zaux.size(0), -1)

            mse = F.mse_loss(za_flat, zp_flat).item()
            cos = F.cosine_similarity(za_flat, zp_flat, dim=1).mean().item()

            all_mse.append(mse)
            all_cos.append(cos)

            batch_n += 1
            if batch_n >= n_batches:
                break

        avg_mse = sum(all_mse) / len(all_mse)
        avg_cos = sum(all_cos) / len(all_cos)

        print(f"\n  Substitute Client Quality (vs victim client):")
        print(f"    MSE              : {avg_mse:.4f}  (lower=better)")
        print(f"    Cosine Similarity: {avg_cos:.4f}  (higher=better, 1.0=identical)")

        return {"mse": avg_mse, "cosine_similarity": avg_cos}

    # ──────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all FORA networks and history."""
        torch.save({
            "substitute":         self.substitute.state_dict(),
            "discriminator":      self.discriminator.state_dict(),
            "inverse_net":        self.inverse_net.state_dict(),
            "opt_substitute":     self.opt_substitute.state_dict(),
            "opt_discriminator":  self.opt_discriminator.state_dict(),
            "opt_inverse":        self.opt_inverse.state_dict(),
            "history":            self.history,
            "smashed_channels":   self.smashed_channels,
            "smashed_spatial":    self.smashed_spatial,
            "cut_layer":          self.cut_layer,
        }, path)
        print(f"[FORA] Saved to {path}")

    def load(self, path: str) -> None:
        """Load FORA networks from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.substitute.load_state_dict(ckpt["substitute"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.inverse_net.load_state_dict(ckpt["inverse_net"])
        self.opt_substitute.load_state_dict(ckpt["opt_substitute"])
        self.opt_discriminator.load_state_dict(ckpt["opt_discriminator"])
        self.opt_inverse.load_state_dict(ckpt["opt_inverse"])
        self.history = ckpt["history"]
        print(f"[FORA] Loaded from {path}")