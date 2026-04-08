"""
defenses/differential_privacy.py
──────────────────────────────────
Differential Privacy Defense — Activation Perturbation at Cut Layer

Threat model addressed:
    Semi-honest server attempting smashed-data reconstruction (Inverse Network,
    UnSplit, PCAT, FORA). The server follows the protocol but passively inverts
    the smashed data it legitimately receives.

Mechanism:
    The CLIENT adds calibrated noise to smashed data BEFORE transmitting to the
    server. This is "activation perturbation DP" — distinct from DP-SGD (which
    perturbs gradients) and from TPSL (Yang et al., 2022, which perturbs
    gradients to protect labels in vertical FL).

    For reconstruction defense, the smashed data itself is what the server
    inverts, so the noise must corrupt that tensor directly.

    Formally, for smashed data z = f_client(x):

        Gaussian (ε,δ)-DP:   z̃ = clip(z, C) + N(0, σ²C²·I)
        Laplace  (ε,0)-DP:   z̃ = clip(z, C) + Lap(C/ε)

    where C is the sensitivity (clipping norm), σ is the noise multiplier
    chosen so that σ ≥ √(2 ln(1.25/δ)) / ε for Gaussian mechanism.

    The noisy z̃ is then sent to the server and treated as normal smashed data.
    The server never knows whether noise was applied.

Why clip before adding noise?
    The DP guarantee requires bounded sensitivity — the maximum change in
    smashed data from changing one training sample. Clipping to norm C gives
    a sensitivity of C, so the noise calibration is well-defined. This mirrors
    the gradient clipping step in DP-SGD (Abadi et al., 2016).

    Without clipping, σ would need to be set conservatively large to account
    for arbitrary activation magnitudes, collapsing utility.

Key design choices (informed by Pham et al., 2023):
    - Noise is injected at the cut layer (directly on smashed_data), NOT on
      intermediate layers. This gives the best accuracy-privacy tradeoff.
    - Both Gaussian and Laplace mechanisms are supported for empirical comparison.
    - Noise is applied only during training, not at inference (can be toggled).
    - The same interface as NoPeekNNTrainer is maintained for drop-in use in
      evaluate_attack_defense.py.

Privacy accounting:
    Naïve composition: ε_total = ε × (number of training steps)
    This grows linearly with training, so ε here is the per-step budget.
    For tighter bounds, Moments Accountant / Rényi DP could be added,
    but naïve composition is sufficient for benchmarking comparisons.

References:
    - Abadi et al. (2016) "Deep Learning with Differential Privacy" (DP-SGD)
    - Pham et al. (2023) "Enhancing Accuracy-Privacy Trade-off in Differentially
      Private Split Learning" arXiv:2310.14434
    - Yang et al. (2022) "TPSL: Differentially Private Label Protection in
      Split Learning" (gradient perturbation — different mechanism)
    - Gawron & Stubbings (2022) "Feature Space Hijacking Attacks against
      Differentially Private Split Learning" (shows DP vs FSHA)

Expected results (calibrated to proposal targets):
    Noise level      | Test Acc | SSIM vs baseline | Practical?
    ─────────────────|──────────|──────────────────|───────────
    σ=0.01 (high ε)  |  ~84%    |  marginal drop   | Yes
    σ=0.1  (mid ε)   |  ~80%    |  moderate drop   | Yes
    σ=1.0  (low ε)   |  ~70%    |  large drop      | Borderline
    σ=5.0  (tiny ε)  |  <60%    |  major drop      | No
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.reconstruction import distance_correlation


# ─────────────────────────────────────────────────────────────────────────────
# Noise mechanism functions (standalone, also usable outside the trainer)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_noise(
    z: torch.Tensor,
    sensitivity: float,
    noise_multiplier: float,
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise for (ε,δ)-DP.

    Noise scale: σ_actual = noise_multiplier × sensitivity
    Privacy:     ε ≈ √(2 ln(1.25/δ)) / noise_multiplier  (Gaussian mechanism)

    Args:
        z:               Smashed data tensor (any shape)
        sensitivity:     L2 clipping norm C — upper bound on ‖z‖₂ per sample
        noise_multiplier: σ / C ratio; larger = more privacy, less utility

    Returns:
        Perturbed tensor z̃ = z + N(0, (noise_multiplier × sensitivity)²·I)
    """
    std = noise_multiplier * sensitivity
    return z + torch.randn_like(z) * std


def laplace_noise(
    z: torch.Tensor,
    sensitivity: float,
    epsilon: float,
) -> torch.Tensor:
    """
    Add calibrated Laplace noise for (ε,0)-DP.

    Noise scale: b = sensitivity / ε
    Privacy:     ε-DP (pure differential privacy, δ=0)

    Args:
        z:           Smashed data tensor (any shape)
        sensitivity: L1/L2 sensitivity (clipping norm)
        epsilon:     Privacy budget ε; smaller = stronger privacy

    Returns:
        Perturbed tensor z̃ = z + Lap(sensitivity / ε)
    """
    scale = sensitivity / epsilon
    return z + torch.distributions.Laplace(0, scale).sample(z.shape).to(z.device)


def clip_per_sample(z: torch.Tensor, clip_norm: float) -> torch.Tensor:
    """
    Clip each sample in a batch to have L2 norm ≤ clip_norm.

    This bounds the sensitivity of the smashed data, which is required to
    calibrate the noise correctly. Applied per-sample (not per-batch) to
    match the standard DP-SGD clipping operation.

    Args:
        z:         Smashed data (B, C, H, W) or (B, D)
        clip_norm: Maximum L2 norm per sample

    Returns:
        Clipped smashed data with ‖z[i]‖₂ ≤ clip_norm for all i
    """
    b = z.shape[0]
    z_flat = z.view(b, -1)
    norms = z_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale = (clip_norm / norms).clamp(max=1.0)
    return (z_flat * scale).view(z.shape)


def compute_epsilon(
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    """
    Compute ε for a single-step Gaussian mechanism given σ/C and δ.

    ε = √(2 ln(1.25/δ)) / noise_multiplier

    Args:
        noise_multiplier: σ / C ratio
        delta:            δ for (ε,δ)-DP (default 1e-5)

    Returns:
        Per-step ε
    """
    if noise_multiplier <= 0:
        return float("inf")
    return math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier


# ─────────────────────────────────────────────────────────────────────────────
# DP Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DifferentialPrivacyTrainer:
    """
    Vanilla Split Learning trainer augmented with Differential Privacy noise
    injected at the cut layer (activation perturbation).

    Two noise mechanisms are supported:
        'gaussian'  → (ε,δ)-DP via Gaussian mechanism (most common in practice)
        'laplace'   → (ε,0)-DP via Laplace mechanism (pure DP, stronger guarantee)

    The interface mirrors NoPeekNNTrainer for drop-in use in the evaluation
    pipeline. The only behavioral difference is in train_step(), where smashed
    data is clipped and noised before being passed to the server.

    Training-time vs inference-time noise:
        By default, noise is applied ONLY during training (apply_noise_at_eval=False).
        This is the standard setting: the defense protects smashed data during the
        training protocol. At inference, the deployed model can run cleanly.
        Set apply_noise_at_eval=True to also protect inference-time smashed data.
    """

    def __init__(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        # Noise config
        mechanism: str = "gaussian",          # 'gaussian' or 'laplace'
        noise_multiplier: float = 0.1,        # σ/C for Gaussian; ignored for Laplace
        epsilon: float = 1.0,                 # ε for Laplace; used for reporting with Gaussian
        delta: float = 1e-5,                  # δ for Gaussian (ε,δ)-DP
        clip_norm: float = 1.0,               # Sensitivity clipping norm C
        apply_noise_at_eval: bool = False,    # Add noise during evaluation too
        # Training config
        lr: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Args:
            client_model:        Client-side model (bottom layers)
            server_model:        Server-side model (top layers + classification)
            train_loader:        Training data loader
            test_loader:         Test data loader
            mechanism:           Noise mechanism: 'gaussian' or 'laplace'
            noise_multiplier:    Gaussian: σ/C ratio. Higher = more noise = more privacy.
                                 Suggested values: 0.01 (weak), 0.1 (moderate), 1.0 (strong)
                                 Ignored for Laplace.
            epsilon:             Laplace: privacy budget ε. Smaller = more privacy.
                                 Gaussian: used only for reporting (ε derived from σ/C).
                                 Suggested: 0.1 (strong), 1.0 (moderate), 10.0 (weak)
            delta:               Gaussian: δ for (ε,δ)-DP. Typically 1e-5.
            clip_norm:           L2 clipping norm C. Controls sensitivity bound.
                                 Affects both clipping distortion and noise scale.
                                 Start with 1.0; reduce if norms are typically smaller.
            apply_noise_at_eval: If True, noise is added during evaluation too.
                                 Default False (cleaner inference).
            lr:                  Learning rate
            device:              Compute device (auto-detected if None)
        """
        # ── Validate mechanism ─────────────────────────────────────────────────
        if mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"mechanism must be 'gaussian' or 'laplace', got '{mechanism}'")

        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # ── Models ────────────────────────────────────────────────────────────
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader = train_loader
        self.test_loader  = test_loader

        # ── DP config ─────────────────────────────────────────────────────────
        self.mechanism           = mechanism
        self.noise_multiplier    = noise_multiplier
        self.epsilon             = epsilon
        self.delta               = delta
        self.clip_norm           = clip_norm
        self.apply_noise_at_eval = apply_noise_at_eval

        # Effective ε per step:
        #   Gaussian: derived from noise_multiplier and delta
        #   Laplace:  directly from epsilon argument
        if mechanism == "gaussian":
            self.effective_epsilon = compute_epsilon(noise_multiplier, delta)
        else:
            self.effective_epsilon = epsilon

        # ── Optimizers ────────────────────────────────────────────────────────
        self.client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
        self.server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "train_loss":        [],
            "train_acc":         [],
            "test_loss":         [],
            "test_acc":          [],
            "epoch_time":        [],
            "smashed_norm_mean": [],  # track pre-clip norms for calibration insight
            "clip_fraction":     [],  # fraction of samples clipped each epoch
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Noise injection (the core defense operation)
    # ──────────────────────────────────────────────────────────────────────────

    def _perturb_smashed(
        self,
        smashed: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Apply sensitivity clipping and calibrated noise to smashed data.

        This is the entire defense in one operation:
            1. Clip per-sample norms to C (bound sensitivity)
            2. Add noise calibrated to C (Gaussian or Laplace)

        Args:
            smashed:  Raw smashed data from client model (B, ...)
            training: If False and apply_noise_at_eval is False, skip noise

        Returns:
            (perturbed_smashed, mean_norm_before_clip, clip_fraction)
        """
        with torch.no_grad():
            # Track norms before clipping (for diagnostics)
            b = smashed.shape[0]
            norms = smashed.view(b, -1).norm(dim=1)
            mean_norm = norms.mean().item()
            clip_fraction = (norms > self.clip_norm).float().mean().item()

        if not training and not self.apply_noise_at_eval:
            return smashed, mean_norm, clip_fraction

        # Step 1: Clip to bound sensitivity
        clipped = clip_per_sample(smashed, self.clip_norm)

        # Step 2: Add calibrated noise
        if self.mechanism == "gaussian":
            perturbed = gaussian_noise(clipped, self.clip_norm, self.noise_multiplier)
        else:  # laplace
            perturbed = laplace_noise(clipped, self.clip_norm, self.epsilon)

        return perturbed, mean_norm, clip_fraction

    # ──────────────────────────────────────────────────────────────────────────
    # Core training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        One DP split learning training step.

        Protocol:
            1. Client forward → smashed_data
            2. CLIENT clips smashed_data norms + adds DP noise → smashed_noisy
               (this step is invisible to the server)
            3. Simulate network boundary: detach smashed_noisy
            4. Server forward on smashed_noisy → loss → backward
            5. Grad flows back → client backward (through noisy smashed_data)

        Note on gradient flow:
            The noise is added INSIDE the client's computation graph
            (before detach). This means the server's gradient flows back
            through the noise operation during client backprop. The noise
            itself has zero gradient w.r.t. model parameters, so this is
            equivalent to adding a fixed perturbation from the gradient's
            perspective — it degrades the signal but doesn't block learning.

        Args:
            images: Batch of input images
            labels: Batch of ground-truth labels

        Returns:
            (loss, accuracy, mean_pre_clip_norm, clip_fraction)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        # ── Client forward ────────────────────────────────────────────────────
        self.client_optimizer.zero_grad()
        smashed_data = self.client_model(images)

        # ── DP perturbation (client-side, before "transmission") ───────────────
        # We add noise inside autograd so that:
        #   a) The server receives noisy smashed data (privacy protection)
        #   b) Gradients from the server flow back through the noisy tensor
        #      to update the client model (training continues)
        smashed_noisy, mean_norm, clip_frac = self._perturb_smashed(
            smashed_data, training=True
        )

        # ── Simulate network boundary ─────────────────────────────────────────
        # The server receives smashed_noisy — it cannot distinguish noise from
        # legitimate activations.
        smashed_for_server = smashed_noisy.detach().requires_grad_(True)

        # ── Server forward + backward ─────────────────────────────────────────
        self.server_optimizer.zero_grad()
        outputs   = self.server_model(smashed_for_server)
        loss      = self.criterion(outputs, labels)
        loss.backward()

        grad_to_client = smashed_for_server.grad.clone()
        self.server_optimizer.step()

        # ── Client backward ───────────────────────────────────────────────────
        # Gradient is w.r.t. smashed_noisy. The noise has no parameters,
        # so the gradient passes through cleanly (straight-through).
        smashed_noisy.backward(grad_to_client)
        self.client_optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────────
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).sum().item() / labels.size(0)

        return loss.item(), accuracy, mean_norm, clip_frac

    # ──────────────────────────────────────────────────────────────────────────
    # Epoch-level loops
    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> Tuple[float, float, float, float]:
        """Train one epoch. Returns (avg_loss, avg_acc, avg_norm, avg_clip_frac)."""
        self.client_model.train()
        self.server_model.train()

        total_loss     = 0.0
        total_acc      = 0.0
        total_norm     = 0.0
        total_clip_frac = 0.0
        total_samples  = 0
        n_batches      = 0

        pbar = tqdm(self.train_loader, desc=f"Training (DP-{self.mechanism})", leave=False)
        for images, labels in pbar:
            loss, acc, norm, clip_frac = self.train_step(images, labels)

            bs = labels.size(0)
            total_loss     += loss     * bs
            total_acc      += acc      * bs
            total_samples  += bs
            total_norm     += norm
            total_clip_frac += clip_frac
            n_batches      += 1

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "acc":  f"{acc:.3f}",
                "norm": f"{norm:.2f}",
            })

        return (
            total_loss      / total_samples,
            total_acc       / total_samples,
            total_norm      / n_batches,
            total_clip_frac / n_batches,
        )

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on test set. Returns (avg_loss, avg_accuracy)."""
        self.client_model.eval()
        self.server_model.eval()

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            smashed = self.client_model(images)

            # Optionally add noise at eval time too
            if self.apply_noise_at_eval:
                smashed, _, _ = self._perturb_smashed(smashed, training=True)

            outputs = self.server_model(smashed)
            loss    = self.criterion(outputs, labels)

            total_loss    += loss.item() * labels.size(0)
            _, predicted   = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    # ──────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, epochs: int, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train with DP defense.

        Args:
            epochs:  Number of training epochs
            verbose: Print per-epoch metrics

        Returns:
            Training history
        """
        n_client = sum(p.numel() for p in self.client_model.parameters())
        n_server = sum(p.numel() for p in self.server_model.parameters())

        # ── Banner ────────────────────────────────────────────────────────────
        print("\nStarting Differential Privacy Defense Training")
        print("=" * 65)
        print(f"  Client model parameters : {n_client:>10,}")
        print(f"  Server model parameters : {n_server:>10,}")
        print(f"  Mechanism               : {self.mechanism}")
        print(f"  Clipping norm (C)       : {self.clip_norm}")

        if self.mechanism == "gaussian":
            print(f"  Noise multiplier (σ/C)  : {self.noise_multiplier}")
            print(f"  δ                       : {self.delta}")
            print(f"  Per-step ε (approx)     : {self.effective_epsilon:.4f}")
            print(f"  Noise std (σ = σ/C × C) : {self.noise_multiplier * self.clip_norm:.4f}")
        else:
            print(f"  ε (Laplace)             : {self.epsilon}")
            print(f"  Laplace scale (C/ε)     : {self.clip_norm / self.epsilon:.4f}")

        print(f"  Noise at eval           : {self.apply_noise_at_eval}")
        print("=" * 65)
        print("  Defense: smashed_data → clip(z, C) + noise → server")
        print("=" * 65 + "\n")

        for epoch in range(epochs):
            t0 = time.time()

            train_loss, train_acc, mean_norm, clip_frac = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            epoch_time = time.time() - t0

            # Record
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["epoch_time"].append(epoch_time)
            self.history["smashed_norm_mean"].append(mean_norm)
            self.history["clip_fraction"].append(clip_frac)

            if verbose:
                clip_pct = clip_frac * 100
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc*100:5.2f}% | "
                    f"Test Acc: {test_acc*100:5.2f}% | "
                    f"Norm: {mean_norm:.2f} | "
                    f"Clipped: {clip_pct:.1f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("Differential Privacy Training Complete!")
        print(f"  Best Test Accuracy      : {max(self.history['test_acc'])*100:.2f}%")
        print(f"  Avg Clipped Samples     : {sum(self.history['clip_fraction'])/len(self.history['clip_fraction'])*100:.1f}%")
        if self.mechanism == "gaussian":
            print(f"  Per-step ε              : {self.effective_epsilon:.4f}")
            print(f"  Naïve composition ε     : {self.effective_epsilon * epochs:.2f}  "
                  f"(over {epochs} epochs; use Rényi DP for tighter bound)")
        print("=" * 65)

        return self.history

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def get_smashed_data(self, images: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        """
        Get smashed data for a batch — useful for downstream attack evaluation.

        Args:
            images: Input images
            noisy:  If True, return perturbed smashed data (what server actually sees)
                    If False, return clean smashed data (for clean evaluation)
        """
        self.client_model.eval()
        with torch.no_grad():
            smashed = self.client_model(images.to(self.device))
            if noisy:
                smashed, _, _ = self._perturb_smashed(smashed, training=True)
            return smashed

    def privacy_budget_report(self, n_steps: int) -> Dict[str, float]:
        """
        Compute and print a privacy budget report for a given number of steps.

        Args:
            n_steps: Total training steps (epochs × batches_per_epoch)

        Returns:
            Dict with per_step_epsilon, total_epsilon_naive, delta
        """
        report = {
            "mechanism":          self.mechanism,
            "per_step_epsilon":   self.effective_epsilon,
            "total_epsilon_naive": self.effective_epsilon * n_steps,
            "delta":              self.delta if self.mechanism == "gaussian" else 0.0,
            "clip_norm":          self.clip_norm,
        }

        print("\nPrivacy Budget Report")
        print("─" * 40)
        print(f"  Mechanism         : {report['mechanism']}")
        print(f"  Per-step ε        : {report['per_step_epsilon']:.4f}")
        print(f"  Total ε (naïve)   : {report['total_epsilon_naive']:.2f}  ({n_steps} steps)")
        print(f"  δ                 : {report['delta']}")
        print(f"  Sensitivity (C)   : {report['clip_norm']}")
        print("─" * 40)
        print("  Note: Naïve composition gives loose bounds.")
        print("  For publication, use Rényi DP / Moments Accountant.")

        return report

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "client_model":     self.client_model.state_dict(),
            "server_model":     self.server_model.state_dict(),
            "client_optimizer": self.client_optimizer.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
            "mechanism":        self.mechanism,
            "noise_multiplier": self.noise_multiplier,
            "epsilon":          self.epsilon,
            "delta":            self.delta,
            "clip_norm":        self.clip_norm,
            "history":          self.history,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.client_model.load_state_dict(ckpt["client_model"])
        self.server_model.load_state_dict(ckpt["server_model"])
        self.client_optimizer.load_state_dict(ckpt["client_optimizer"])
        self.server_optimizer.load_state_dict(ckpt["server_optimizer"])
        self.mechanism        = ckpt["mechanism"]
        self.noise_multiplier = ckpt["noise_multiplier"]
        self.epsilon          = ckpt["epsilon"]
        self.delta            = ckpt["delta"]
        self.clip_norm        = ckpt["clip_norm"]
        self.history          = ckpt["history"]
        print(f"Checkpoint loaded from {path}")