"""
defenses/nopeeknn.py
─────────────────────
NoPeekNN — Distance Correlation Privacy Defense

Reference:
    Vepakomma et al., "NoPeek: Information leakage reduction to share activations
    in distributed deep learning" (2020)
    https://arxiv.org/abs/2008.03409

Mechanism:
    NoPeekNN adds a distance correlation term to the task loss:

        total_loss = CE(logits, labels) + λ * dCor(smashed_data, raw_input)

    dCor measures statistical dependence between the smashed data and raw input.
    Minimizing dCor during training encourages the client model to produce
    smashed data that is statistically independent of the original input,
    making it harder for an attacker to reconstruct the input.

    λ controls the privacy-utility tradeoff:
        λ = 0.0  → standard split learning (no privacy)
        λ = 0.1  → mild privacy, ~1-2% accuracy cost
        λ = 1.0  → strong privacy pressure, ~2-5% accuracy cost
        λ = 5.0  → aggressive privacy, accuracy degrades significantly

Known limitation (from the proposal):
    FORA bypasses NoPeekNN. The attack learns feature-space behavior rather
    than direct input-output mappings, so statistical independence of smashed
    data from inputs does not prevent FORA's substitute model from working.
    NoPeekNN remains a useful baseline and provides some protection against
    the simpler Inverse Network attack.

Implementation approach:
    NoPeekNN is implemented as a WRAPPER around VanillaSplitTrainer (and
    optionally UShapedSplitTrainer). We override train_step() to inject
    the dCor regularization into the loss before backpropagation.

    This design avoids duplicating the entire training loop and keeps the
    defense orthogonally composable with other components.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.reconstruction import distance_correlation


# ─────────────────────────────────────────────────────────────────────────────
# NoPeekNN Trainer (wraps Vanilla SL)
# ─────────────────────────────────────────────────────────────────────────────

class NoPeekNNTrainer:
    """
    Vanilla Split Learning trainer augmented with the NoPeekNN distance
    correlation regularization.

    Compared to VanillaSplitTrainer, the only change is in train_step:
        loss = criterion(outputs, labels) + λ * dCor(smashed_data, images)

    The dCor term is computed BEFORE detaching the smashed data, so its
    gradients flow back through the client model during the client backward pass.
    This is the key: dCor regularization shapes the CLIENT model's representation.

    dCor subsample:
        dCor computation is O(N²) in batch size. For batches > 64, we subsample
        a random subset to keep the overhead manageable.
    """

    def __init__(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lambda_dcor: float = 0.5,
        dcor_subsample: int = 64,
        lr: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Args:
            client_model:   Client-side model (bottom layers)
            server_model:   Server-side model (top layers + classification)
            train_loader:   Training data loader
            test_loader:    Test data loader
            lambda_dcor:    Weight of the distance correlation penalty.
                            0.0 = no defense, 1.0 = equal weighting with task loss.
            dcor_subsample: Max number of samples to use for dCor computation
                            per batch. Reduces O(N²) cost for large batches.
            lr:             Learning rate
            device:         Compute device (auto-detected if None)
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
        print(f"Using device: {self.device}")

        # ── Models ────────────────────────────────────────────────────────────
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader = train_loader
        self.test_loader  = test_loader

        # ── Defense config ────────────────────────────────────────────────────
        self.lambda_dcor    = lambda_dcor
        self.dcor_subsample = dcor_subsample

        # ── Optimizers ────────────────────────────────────────────────────────
        self.client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
        self.server_optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "train_loss":      [],
            "train_task_loss": [],
            "train_dcor_loss": [],
            "train_acc":       [],
            "test_loss":       [],
            "test_acc":        [],
            "epoch_time":      [],
            "dcor_values":     [],  # raw dCor (privacy leakage level)
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        images = images.to(self.device)
        labels = labels.to(self.device)

        # ── Client forward ────────────────────────────────────────────────────
        self.client_optimizer.zero_grad()
        smashed_data = self.client_model(images)

        # ── Distance correlation (before detach so grad flows to client) ──────
        if self.lambda_dcor > 0:
            n = smashed_data.shape[0]
            if n > self.dcor_subsample:
                idx = torch.randperm(n, device=self.device)[:self.dcor_subsample]
                smashed_sub = smashed_data[idx]
                images_sub  = images[idx]
            else:
                smashed_sub = smashed_data
                images_sub  = images

            dcor_value = distance_correlation(images_sub.detach(), smashed_sub)
            dcor_loss  = self.lambda_dcor * dcor_value

            # ← KEY FIX: dCor backward FIRST with retain_graph=True so the
            # graph through smashed_data stays alive for the second backward below.
            dcor_loss.backward(retain_graph=True)
        else:
            dcor_value = torch.tensor(0.0, device=self.device)

        # ── Simulate network boundary ─────────────────────────────────────────
        smashed_data_server = smashed_data.detach().requires_grad_(True)

        # ── Server forward ────────────────────────────────────────────────────
        self.server_optimizer.zero_grad()
        outputs   = self.server_model(smashed_data_server)
        task_loss = self.criterion(outputs, labels)
        task_loss.backward()

        grad_to_client = smashed_data_server.grad.clone()
        self.server_optimizer.step()

        # ── Client backward: server grad stacked on top of dCor grad ─────────
        # dCor grads are already accumulated in .grad from the backward above.
        # This second backward adds the server's contribution on top.
        smashed_data.backward(grad_to_client)
        self.client_optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────────
        _, predicted = outputs.max(1)
        accuracy     = predicted.eq(labels).sum().item() / labels.size(0)
        total_loss   = task_loss.item() + dcor_value.item() * self.lambda_dcor

        return total_loss, task_loss.item(), dcor_value.item(), accuracy
    # ──────────────────────────────────────────────────────────────────────────
    # Epoch-level loops
    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> Tuple[float, float, float, float]:
        """Train one epoch. Returns (total_loss, task_loss, dcor_loss, accuracy)."""
        self.client_model.train()
        self.server_model.train()

        total_loss_acc  = 0.0
        task_loss_acc   = 0.0
        dcor_acc        = 0.0
        correct_acc     = 0.0
        total_samples   = 0

        pbar = tqdm(self.train_loader, desc="Training (NoPeekNN)", leave=False)
        for images, labels in pbar:
            total, task, dcor, acc = self.train_step(images, labels)

            bs = labels.size(0)
            total_loss_acc += total * bs
            task_loss_acc  += task  * bs
            dcor_acc       += dcor  * bs
            correct_acc    += acc   * bs
            total_samples  += bs

            pbar.set_postfix({
                "task": f"{task:.4f}",
                "dcor": f"{dcor:.4f}",
                "acc":  f"{acc:.3f}",
            })

        n = total_samples
        return (
            total_loss_acc / n,
            task_loss_acc  / n,
            dcor_acc       / n,
            correct_acc    / n,
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
            outputs = self.server_model(smashed)
            loss    = self.criterion(outputs, labels)

            total_loss    += loss.item() * labels.size(0)
            _, predicted   = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    # ──────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, epochs: int, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train with NoPeekNN defense.

        Args:
            epochs:  Number of training epochs
            verbose: Print per-epoch metrics (includes dCor tracking)

        Returns:
            Training history with separated task/privacy loss components
        """
        n_client = sum(p.numel() for p in self.client_model.parameters())
        n_server = sum(p.numel() for p in self.server_model.parameters())

        print("\nStarting NoPeekNN Defense Training")
        print("=" * 60)
        print(f"  Client model parameters  : {n_client:>10,}")
        print(f"  Server model parameters  : {n_server:>10,}")
        print(f"  λ (dcor weight)          : {self.lambda_dcor}")
        print(f"  dCor subsample size      : {self.dcor_subsample}")
        print("=" * 60)
        print("  Loss = CE(logits, labels) + λ * dCor(smashed, input)")
        print("=" * 60 + "\n")

        for epoch in range(epochs):
            t0 = time.time()

            total_loss, task_loss, dcor_val, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            epoch_time = time.time() - t0

            # Record
            self.history["train_loss"].append(total_loss)
            self.history["train_task_loss"].append(task_loss)
            self.history["train_dcor_loss"].append(self.lambda_dcor * dcor_val)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["epoch_time"].append(epoch_time)
            self.history["dcor_values"].append(dcor_val)

            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Total: {total_loss:.4f} | "
                    f"Task: {task_loss:.4f} | "
                    f"dCor: {dcor_val:.4f} | "
                    f"Train Acc: {train_acc*100:5.2f}% | "
                    f"Test Acc: {test_acc*100:5.2f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

        print("\n" + "=" * 60)
        print("NoPeekNN Training Complete!")
        print(f"  Best Test Accuracy    : {max(self.history['test_acc'])*100:.2f}%")
        print(f"  Final dCor (leakage)  : {self.history['dcor_values'][-1]:.4f}")
        print(f"  Initial dCor          : {self.history['dcor_values'][0]:.4f}")
        reduction = self.history["dcor_values"][0] - self.history["dcor_values"][-1]
        print(f"  dCor reduction        : {reduction:.4f}  ({'↓ improved' if reduction > 0 else '↑ worsened'})")
        print("=" * 60)

        return self.history

    def get_smashed_data(self, images: torch.Tensor) -> torch.Tensor:
        """Get smashed data for a batch — useful for downstream attack evaluation."""
        self.client_model.eval()
        with torch.no_grad():
            return self.client_model(images.to(self.device))

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "client_model":     self.client_model.state_dict(),
            "server_model":     self.server_model.state_dict(),
            "client_optimizer": self.client_optimizer.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
            "lambda_dcor":      self.lambda_dcor,
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
        self.lambda_dcor = ckpt["lambda_dcor"]
        self.history     = ckpt["history"]
        print(f"Checkpoint loaded from {path}")