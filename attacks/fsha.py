"""
attacks/fsha.py

Feature-Space Hijacking Attack (FSHA)

Reference:
    Pasquini et al., "Unleashing the Tiger: Inference Attacks on Split Learning"
    CCS 2021 | arXiv:2012.02670

Threat model: MALICIOUS SERVER
    The server deviates from the split learning protocol by sending crafted
    adversarial gradients to the client instead of legitimate task gradients.
    The client unknowingly trains its model (f) to map inputs into a feature
    space Z̃ that the attacker controls and can invertibly decode.

Three attacker-side networks:
    f̃      Pilot encoder  — maps X_pub → Z̃ (same output shape as client model f)
    f̃⁻¹    Inverse decoder — maps Z̃ → X (reconstructs inputs from pilot's space)
    D       Discriminator  — Wasserstein critic distinguishing f̃(X_pub) from f(X_priv)

Training dynamics (WGAN-GP):
    AE loss   : min MSE(f̃⁻¹(f̃(x_pub)), x_pub)
    Disc loss : min E[D(f(x_priv))] - E[D(f̃(x_pub))] + λ_gp * GP   (Wasserstein)
    Adv grad  : ∂/∂smashed (-D(smashed)) → sent to client as "task gradient"
                Forces f to produce smashed data with high D-score (looks like f̃)

After convergence:
    f(X_priv) ≈ Z̃ in distribution
    X̃_priv = f̃⁻¹(f(X_priv)) ≈ X_priv

Architecture note (from Table A.1 of the paper):
    f̃  uses strided convolutions (intentionally different from client's MaxPool)
    f̃⁻¹ uses transposed convolutions
    D   uses residual blocks + Wasserstein (no sigmoid)
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.reconstruction import compute_ssim, compute_psnr


# ─────────────────────────────────────────────────────────────────────────────
# Smashed-data shape registry for SimpleCNN
# ─────────────────────────────────────────────────────────────────────────────

SMASHED_SHAPES = {
    # cut_layer: (channels, height, width)
    1: (32, 16, 16),
    2: (64,  8,  8),
    3: (128, 4,  4),
}


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block for the discriminator D.
    Activation applied BEFORE convolutions (pre-activation style) so that
    the raw linear output of D is unbounded — required for Wasserstein training.
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.shortcut = (
            nn.Conv2d(channels, channels, 1, stride) if stride > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(x, 0.2, inplace=False)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2, inplace=False)
        h = self.conv2(h)
        return h + self.shortcut(x)


# ─────────────────────────────────────────────────────────────────────────────
# f̃  — Pilot network
# ─────────────────────────────────────────────────────────────────────────────

class PilotNetwork(nn.Module):
    """
    f̃: Pilot encoder.

    Maps X_pub ∈ R^(3×32×32) → Z̃ ∈ R^(C_s×H_s×W_s), where (C_s, H_s, W_s)
    matches the smashed-data shape of the victim's client model f.

    Deliberately uses strided convolutions instead of MaxPool so that f̃ has
    a different architecture from f (as required by the threat model — the
    attacker does not know f's internals). Simpler / shallower than f.
    """

    def __init__(
        self,
        out_channels: int,
        out_spatial: int,
        in_channels: int = 3,
        in_spatial: int = 32,
    ):
        super().__init__()
        n_strides = int(math.log2(in_spatial // out_spatial))

        layers: List[nn.Module] = []
        cur_ch = in_channels
        # Increasing channel schedule: keeps f̃ simple while reaching target shape
        ch_schedule = [64, 128, 256, 512]

        for i in range(n_strides):
            next_ch = ch_schedule[min(i, len(ch_schedule) - 1)]
            layers += [
                nn.Conv2d(cur_ch, next_ch, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            cur_ch = next_ch

        # Final 1×1-style projection to exact output channel count
        layers += [
            nn.Conv2d(cur_ch, out_channels, kernel_size=3, stride=1, padding=1),
            # No activation: pilot network output is an unconstrained feature map
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# f̃⁻¹ — Inverse / decoder network
# ─────────────────────────────────────────────────────────────────────────────

class InverseMapper(nn.Module):
    """
    f̃⁻¹: Decoder network.

    Maps Z̃ ∈ R^(C_s×H_s×W_s) → X̃ ∈ R^(3×32×32).
    Trained jointly with f̃ as an autoencoder:
        min  MSE(f̃⁻¹(f̃(x_pub)), x_pub)

    After convergence, this can also invert the VICTIM's outputs because
    f has been forced to map into Z̃ via the adversarial gradient.
    """

    def __init__(
        self,
        in_channels: int,
        in_spatial: int,
        out_channels: int = 3,
        out_spatial: int = 32,
    ):
        super().__init__()
        n_upsample = int(math.log2(out_spatial // in_spatial))

        layers: List[nn.Module] = []
        cur_ch = in_channels

        # Initial refinement conv (stays at same spatial size)
        layers += [
            nn.Conv2d(cur_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        cur_ch = 256

        # Upsampling stages
        for i in range(n_upsample):
            next_ch = max(cur_ch // 2, 64)
            layers += [
                nn.ConvTranspose2d(cur_ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(inplace=True),
            ]
            cur_ch = next_ch

        # Final pixel-level projection → Tanh bounds output to (−1, 1)
        layers += [
            nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# D — Wasserstein discriminator
# ─────────────────────────────────────────────────────────────────────────────

class FSHADiscriminator(nn.Module):
    """
    D: Wasserstein (no-sigmoid) discriminator on smashed data.

    Trained to distinguish f̃(X_pub) [high score / "real"] from
    f(X_priv) [low score / "fake"].

    Residual architecture, deeper than f̃ and f̃⁻¹, to force close
    topological matching between the two feature spaces.
    """

    def __init__(self, in_channels: int, in_spatial: int):
        super().__init__()
        out_spatial = in_spatial // 2   # one stride-2 downsampling

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # ↓2×
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(256 * out_spatial * out_spatial, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.body(z)
        return self.fc(h.view(h.size(0), -1))


# ─────────────────────────────────────────────────────────────────────────────
# WGAN-GP helper
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float,
) -> torch.Tensor:
    """
    Wasserstein gradient penalty (Gulrajani et al., 2017).

    Enforces the 1-Lipschitz constraint on D by penalising the gradient norm
    at random convex interpolations of real and fake samples.
    """
    # Align batch sizes
    n = min(real.size(0), fake.size(0))
    real, fake = real[:n], fake[:n]

    alpha = torch.rand(n, 1, 1, 1, device=device)
    interp = (alpha * real.detach() + (1.0 - alpha) * fake.detach()).requires_grad_(True)

    d_interp = D(interp)

    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Penalise deviation of gradient norm from 1
    gp = ((grads.norm(2, dim=(1, 2, 3)) - 1.0) ** 2).mean()
    return lambda_gp * gp


# ─────────────────────────────────────────────────────────────────────────────
# Main FSHA class
# ─────────────────────────────────────────────────────────────────────────────

class FSHAAttack:
    """
    Feature-Space Hijacking Attack.

    Usage
    ─────
        # Build attack (attacker controls the server)
        fsha = FSHAAttack(
            cut_layer=2,
            public_loader=test_loader,   # server's auxiliary X_pub (same domain)
        )

        # Hijack training: call run_hijacked_training() or drive the loop manually
        history = fsha.run_hijacked_training(
            client_model=client,
            client_optimizer=client_opt,
            private_loader=train_loader,
            n_setup_iters=3000,
        )

        # Reconstruct private data at inference time
        images = next(iter(train_loader))[0].to(device)
        smashed = client_model(images)
        reconstructed = fsha.reconstruct(smashed)   # X̃_priv = f̃⁻¹(f(X_priv))
    """

    def __init__(
        self,
        cut_layer: int,
        public_loader: DataLoader,
        lr_ae:    float = 1e-4,      # Learning rate for f̃ + f̃⁻¹
        lr_disc:  float = 1e-4,      # Learning rate for D  (10× higher for stability)
        lambda_gp: float = 500.0,    # Gradient-penalty weight (paper Table B.2)
        n_disc_steps: int = 3,       # D update steps per generator step
        device: Optional[str] = None,
    ):
        """
        Args:
            cut_layer:     Which cut layer is used in the victim's split learning
            public_loader: DataLoader for attacker's auxiliary data X_pub.
                           Must be from the SAME DOMAIN as X_priv but no overlap required.
            lr_ae:         Adam LR for f̃ and f̃⁻¹  (paper: 1e-5; 1e-4 speeds up CIFAR)
            lr_disc:       Adam LR for discriminator D
            lambda_gp:     WGAN-GP gradient penalty coefficient (paper: 500)
            n_disc_steps:  How many D updates per autoencoder/adversarial update
            device:        Compute device (auto-detected if None)
        """
        if cut_layer not in SMASHED_SHAPES:
            raise ValueError(f"cut_layer must be 1, 2, or 3. Got {cut_layer}.")

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.cut_layer   = cut_layer
        self.public_loader = public_loader
        self.lambda_gp   = lambda_gp
        self.n_disc_steps = n_disc_steps

        s_ch, s_h, s_w = SMASHED_SHAPES[cut_layer]

        # ── Attacker's three networks ──────────────────────────────────────────
        self.f_tilde     = PilotNetwork(out_channels=s_ch, out_spatial=s_h).to(self.device)
        self.f_tilde_inv = InverseMapper(in_channels=s_ch, in_spatial=s_h).to(self.device)
        self.D           = FSHADiscriminator(in_channels=s_ch, in_spatial=s_h).to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        # AE optimizer covers both f̃ and f̃⁻¹ jointly (autoencoder objective)
        self.opt_ae = torch.optim.Adam(
            list(self.f_tilde.parameters()) + list(self.f_tilde_inv.parameters()),
            lr=lr_ae, betas=(0.5, 0.999),
        )
        self.opt_disc = torch.optim.Adam(
            self.D.parameters(), lr=lr_disc, betas=(0.5, 0.999),
        )

        # ── Infinite public-data iterator ──────────────────────────────────────
        self._pub_iter: Optional[iter] = None

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "ae_loss": [], "disc_loss": [], "adv_loss": [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _next_pub_batch(self) -> torch.Tensor:
        """Return next X_pub batch, cycling through the public loader."""
        if self._pub_iter is None:
            self._pub_iter = iter(self.public_loader)
        try:
            x, _ = next(self._pub_iter)
        except StopIteration:
            self._pub_iter = iter(self.public_loader)
            x, _ = next(self._pub_iter)
        return x.to(self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Core attack step
    # ──────────────────────────────────────────────────────────────────────────

    def setup_step(
        self,
        smashed_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        One FSHA setup iteration.

        Given a batch of smashed data f(X_priv) from the victim client:
          1. Update autoencoder f̃ + f̃⁻¹ on X_pub  (AE loss)
          2. Update discriminator D on real/fake     (WGAN-GP loss)
          3. Compute adversarial gradient for client (∂(-D(smashed))/∂smashed)

        The returned adversarial gradient is what the malicious server sends to
        the client as the "task gradient". The client applies it blindly, slowly
        steering f towards Z̃.

        Args:
            smashed_data: Batch tensor f(X_priv) received from client, shape
                          (B, C_s, H_s, W_s). Does NOT need requires_grad.

        Returns:
            (adv_gradient, metrics)
            adv_gradient: same shape as smashed_data — send to client as grad
            metrics:      dict of scalar losses for logging
        """
        smashed = smashed_data.detach().to(self.device)

        # ── 1. Train autoencoder f̃ + f̃⁻¹ ─────────────────────────────────────
        self.f_tilde.train()
        self.f_tilde_inv.train()

        x_pub = self._next_pub_batch()
        self.opt_ae.zero_grad()
        z_pub  = self.f_tilde(x_pub)
        x_recon = self.f_tilde_inv(z_pub)
        ae_loss  = F.mse_loss(x_recon, x_pub)
        ae_loss.backward()
        self.opt_ae.step()

        # ── 2. Train discriminator D (n_disc_steps updates) ────────────────────
        self.D.train()
        disc_loss_total = 0.0

        for _ in range(self.n_disc_steps):
            x_pub_d = self._next_pub_batch()
            self.opt_disc.zero_grad()

            with torch.no_grad():
                z_real = self.f_tilde(x_pub_d)   # f̃(X_pub) → "real" for D
            z_fake = smashed.detach()              # f(X_priv) → "fake" for D

            # WGAN: max E[D(real)] - E[D(fake)]  ≡  min E[D(fake)] - E[D(real)]
            d_real = self.D(z_real)
            d_fake = self.D(z_fake)

            gp = _gradient_penalty(self.D, z_real, z_fake, self.device, self.lambda_gp)
            disc_loss = d_fake.mean() - d_real.mean() + gp

            disc_loss.backward()
            self.opt_disc.step()
            disc_loss_total += disc_loss.item()

        avg_disc_loss = disc_loss_total / self.n_disc_steps

        # ── 3. Compute adversarial gradient for client ─────────────────────────
        # Client adversarial objective: L_f = -D(f(X_priv))
        # Minimising L_f pushes D(smashed) high → smashed looks like f̃(X_pub)
        # Gradient: ∂L_f/∂smashed = ∂(-D(smashed))/∂smashed
        self.D.eval()
        smashed_grad = smashed.detach().requires_grad_(True)
        adv_loss = -self.D(smashed_grad).mean()
        adv_loss.backward()

        adv_gradient = smashed_grad.grad.clone()

        metrics = {
            "ae_loss":   ae_loss.item(),
            "disc_loss": avg_disc_loss,
            "adv_loss":  adv_loss.item(),
        }

        return adv_gradient, metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Inference (reconstruction)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def reconstruct(self, smashed_data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct private inputs from smashed data.

        X̃_priv = f̃⁻¹(f(X_priv))

        Args:
            smashed_data: Output of the (hijacked) client model, shape (B,C,H,W)

        Returns:
            Reconstructed images, shape (B, 3, 32, 32) — same as original input
        """
        self.f_tilde_inv.eval()
        return self.f_tilde_inv(smashed_data.to(self.device))

    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        client_model: nn.Module,
        data_loader: DataLoader,
        n_batches: int = 8,
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality (SSIM + PSNR) on a data loader.

        Args:
            client_model: Victim's (hijacked) client model
            data_loader:  DataLoader of original private images
            n_batches:    How many batches to use (trims evaluation cost)

        Returns:
            Dict with keys "ssim" and "psnr"
        """
        client_model.eval()
        self.f_tilde_inv.eval()

        ssim_vals, psnr_vals = [], []

        for i, (images, _) in enumerate(data_loader):
            if i >= n_batches:
                break
            images  = images.to(self.device)
            smashed = client_model(images)
            recon   = self.reconstruct(smashed)

            ssim_vals.append(compute_ssim(recon, images).item())
            psnr_vals.append(compute_psnr(recon, images).item())

        return {
            "ssim": sum(ssim_vals) / len(ssim_vals),
            "psnr": sum(psnr_vals) / len(psnr_vals),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Full hijacked training loop
    # ──────────────────────────────────────────────────────────────────────────

    def run_hijacked_training(
        self,
        client_model: nn.Module,
        client_optimizer: torch.optim.Optimizer,
        private_loader: DataLoader,
        n_setup_iters: int = 3000,
        eval_interval: int = 500,
        eval_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """
        Run the full FSHA hijacked training protocol.

        From the client's perspective, this looks identical to normal split
        learning — the client sends smashed data and receives a gradient.
        The difference: the returned gradient is adversarial (from D), not
        from the task loss.

        Args:
            client_model:     Victim's client-side model (will be hijacked in-place)
            client_optimizer: Victim's optimizer (e.g. Adam)
            private_loader:   DataLoader for private training data
            n_setup_iters:    Total number of hijacking iterations
            eval_interval:    Reconstruct and measure SSIM/PSNR every N iters
            eval_loader:      Loader for evaluation; uses private_loader if None
            verbose:          Print progress to stdout

        Returns:
            History dict: ae_loss, disc_loss, adv_loss, ssim, psnr, iter
        """
        client_model = client_model.to(self.device)
        eval_loader  = eval_loader or private_loader

        history: Dict[str, List] = {
            "ae_loss": [], "disc_loss": [], "adv_loss": [],
            "ssim": [], "psnr": [], "iter": [],
        }

        # ── Print summary ──────────────────────────────────────────────────────
        n_victim  = sum(p.numel() for p in client_model.parameters())
        n_pilot   = sum(p.numel() for p in self.f_tilde.parameters())
        n_inv     = sum(p.numel() for p in self.f_tilde_inv.parameters())
        n_disc    = sum(p.numel() for p in self.D.parameters())

        print("\n" + "=" * 65)
        print("  Feature-Space Hijacking Attack (FSHA) — Pasquini et al. 2021")
        print("=" * 65)
        print(f"  Victim client params  : {n_victim:>12,}")
        print(f"  Pilot f̃ params         : {n_pilot:>12,}")
        print(f"  Inverse f̃⁻¹ params     : {n_inv:>12,}")
        print(f"  Discriminator D params : {n_disc:>12,}")
        print(f"  Cut layer              : {self.cut_layer}")
        print(f"  Smashed data shape     : {SMASHED_SHAPES[self.cut_layer]}")
        print(f"  Setup iterations       : {n_setup_iters:>12,}")
        print(f"  λ_gp (gradient pen.)   : {self.lambda_gp:>12.1f}")
        print(f"  D update steps / iter  : {self.n_disc_steps:>12d}")
        print("=" * 65 + "\n")

        priv_iter = iter(private_loader)
        t0 = time.time()

        with tqdm(total=n_setup_iters, desc="FSHA hijacking", unit="iter") as pbar:
            for iteration in range(1, n_setup_iters + 1):

                # Get next private batch
                try:
                    images, _ = next(priv_iter)
                except StopIteration:
                    priv_iter = iter(private_loader)
                    images, _ = next(priv_iter)
                images = images.to(self.device)

                # ── Client forward (victim unaware of attack) ──────────────────
                client_model.train()
                client_optimizer.zero_grad()
                smashed = client_model(images)

                # ── Server: compute adversarial gradient ───────────────────────
                adv_grad, metrics = self.setup_step(smashed)

                # ── Client backward with ADVERSARIAL gradient (hijacked!) ──────
                smashed.backward(adv_grad)
                client_optimizer.step()

                # Track losses
                history["ae_loss"].append(metrics["ae_loss"])
                history["disc_loss"].append(metrics["disc_loss"])
                history["adv_loss"].append(metrics["adv_loss"])

                pbar.update(1)
                pbar.set_postfix({
                    "ae": f"{metrics['ae_loss']:.4f}",
                    "disc": f"{metrics['disc_loss']:.3f}",
                })

                # ── Periodic reconstruction evaluation ─────────────────────────
                if iteration % eval_interval == 0 or iteration == n_setup_iters:
                    eval_m = self.evaluate_reconstruction(client_model, eval_loader)
                    history["ssim"].append(eval_m["ssim"])
                    history["psnr"].append(eval_m["psnr"])
                    history["iter"].append(iteration)

                    if verbose:
                        elapsed = time.time() - t0
                        tqdm.write(
                            f"  Iter {iteration:5d}/{n_setup_iters}"
                            f" | AE: {metrics['ae_loss']:.5f}"
                            f" | Disc: {metrics['disc_loss']:.4f}"
                            f" | SSIM: {eval_m['ssim']:.4f}"
                            f" | PSNR: {eval_m['psnr']:.2f} dB"
                            f" | {elapsed:.0f}s"
                        )

        print("\n" + "=" * 65)
        print("FSHA Setup Complete!")
        if history["ssim"]:
            print(f"  Final SSIM : {history['ssim'][-1]:.4f}")
            print(f"  Final PSNR : {history['psnr'][-1]:.2f} dB")
        print("=" * 65)

        return history

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save attacker-side networks and history."""
        torch.save({
            "f_tilde":     self.f_tilde.state_dict(),
            "f_tilde_inv": self.f_tilde_inv.state_dict(),
            "D":           self.D.state_dict(),
            "history":     self.history,
            "cut_layer":   self.cut_layer,
        }, path)
        print(f"[FSHA] Attacker networks saved → {path}")

    def load(self, path: str):
        """Load previously saved attacker-side networks."""
        ckpt = torch.load(path, map_location=self.device)
        self.f_tilde.load_state_dict(ckpt["f_tilde"])
        self.f_tilde_inv.load_state_dict(ckpt["f_tilde_inv"])
        self.D.load_state_dict(ckpt["D"])
        self.history = ckpt.get("history", self.history)
        print(f"[FSHA] Attacker networks loaded ← {path}")