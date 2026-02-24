"""
U-Shaped Split Learning Trainer

Implements the U-shaped split learning protocol:

    1.  [Client]  Forward through bottom layers → smashed_data
    2.  [Client → Server] Transmit smashed_data  (detach to simulate network boundary)
    3.  [Server]  Forward through middle layers  → server_output
    4.  [Server → Client] Transmit server_output (detach to simulate network boundary)
    5.  [Client]  Forward through top layers     → logits
    6.  [Client]  Compute loss (labels NEVER leave client)
    7.  [Client]  Backward through top layers    → grad_to_server
    8.  [Client → Server] Transmit grad_to_server
    9.  [Server]  Backward through middle layers → grad_to_client
    10. [Server → Client] Transmit grad_to_client
    11. [Client]  Backward through bottom layers

Privacy guarantee: The server never sees raw inputs or labels.
                   It only processes intermediate feature maps.

Gradient flow diagram:
    loss ──► client_top ──► server_middle ──► client_bottom
                        ▲                ▲
                grad_to_server   grad_to_client
                (sent server)    (sent client)
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ushaped import UShapedClientModel


class UShapedSplitTrainer:
    """
    Trainer for U-shaped split learning.

    Unlike vanilla split learning, the loss is computed on the client side
    because the top layers live on the client. This eliminates the need to
    share labels with the server, providing stronger privacy guarantees.
    """

    def __init__(
        self,
        client_model: UShapedClientModel,
        server_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Args:
            client_model: UShapedClientModel (holds bottom + top layers)
            server_model: Middle segment running on server
            train_loader: Training data loader
            test_loader:  Test data loader
            lr:           Learning rate (same for all optimizers)
            device:       'cuda', 'mps', 'cpu', or None for auto-detect
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
        self.test_loader = test_loader

        # ── Optimizers ────────────────────────────────────────────────────────
        # Three separate optimizers matching the three model segments.
        # client_bottom and client_top could share one optimizer, but keeping
        # them separate makes it easy to freeze/unfreeze each segment later.
        self.bottom_optimizer = torch.optim.Adam(
            self.client_model.bottom_parameters(), lr=lr
        )
        self.server_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=lr
        )
        self.top_optimizer = torch.optim.Adam(
            self.client_model.top_parameters(), lr=lr
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "epoch_time": [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        One U-shaped split learning training step.

        The U-shape means labels never reach the server:
            client_bottom → [NETWORK] → server → [NETWORK] → client_top → loss

        Args:
            images: Batch of input images
            labels: Batch of ground-truth labels

        Returns:
            Tuple of (loss_value, accuracy)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        # ── Zero all gradients ────────────────────────────────────────────────
        self.bottom_optimizer.zero_grad()
        self.server_optimizer.zero_grad()
        self.top_optimizer.zero_grad()

        # ── Step 1: Client bottom forward ─────────────────────────────────────
        smashed_data = self.client_model.forward_bottom(images)

        # ── Step 2: Simulate network boundary (client → server) ──────────────
        # detach() breaks the graph; requires_grad_(True) lets server's backward
        # accumulate gradients that we'll later retrieve and hand back to the client.
        smashed_data_server = smashed_data.detach().requires_grad_(True)

        # ── Step 3: Server forward ────────────────────────────────────────────
        server_output = self.server_model(smashed_data_server)

        # ── Step 4: Simulate network boundary (server → client) ──────────────
        server_output_client = server_output.detach().requires_grad_(True)

        # ── Step 5: Client top forward ────────────────────────────────────────
        logits = self.client_model.forward_top(server_output_client)

        # ── Step 6: Loss on client (labels NEVER leave client) ────────────────
        loss = self.criterion(logits, labels)

        # ── Step 7: Client top backward ───────────────────────────────────────
        loss.backward()
        self.top_optimizer.step()

        # ── Step 8: Retrieve grad and send back to server ─────────────────────
        grad_to_server = server_output_client.grad.clone()

        # ── Step 9: Server backward ───────────────────────────────────────────
        server_output.backward(grad_to_server)
        self.server_optimizer.step()

        # ── Step 10: Retrieve grad and send back to client ────────────────────
        grad_to_client = smashed_data_server.grad.clone()

        # ── Step 11: Client bottom backward ──────────────────────────────────
        smashed_data.backward(grad_to_client)
        self.bottom_optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────────
        _, predicted = logits.max(1)
        accuracy = predicted.eq(labels).sum().item() / labels.size(0)

        return loss.item(), accuracy

    # ──────────────────────────────────────────────────────────────────────────
    # Epoch-level training and evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, avg_accuracy)."""
        self.client_model.train()
        self.server_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            loss, acc = self.train_step(images, labels)

            batch_size = labels.size(0)
            total_loss += loss * batch_size
            total_correct += acc * batch_size
            total_samples += batch_size

            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on test set. Returns (avg_loss, avg_accuracy)."""
        self.client_model.eval()
        self.server_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Clean forward pass: bottom → server → top
            smashed = self.client_model.forward_bottom(images)
            server_out = self.server_model(smashed)
            logits = self.client_model.forward_top(server_out)

            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    # ──────────────────────────────────────────────────────────────────────────
    # Top-level training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, epochs: int, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the U-shaped split learning model.

        Args:
            epochs:  Number of epochs
            verbose: Print per-epoch metrics

        Returns:
            Training history dictionary
        """
        n_bottom = sum(p.numel() for p in self.client_model.bottom_parameters())
        n_server = sum(p.numel() for p in self.server_model.parameters())
        n_top    = sum(p.numel() for p in self.client_model.top_parameters())

        print("\nStarting U-Shaped Split Learning Training")
        print("=" * 60)
        print(f"  Client bottom parameters : {n_bottom:>10,}")
        print(f"  Server middle parameters : {n_server:>10,}")
        print(f"  Client top   parameters  : {n_top:>10,}")
        print(f"  Total parameters         : {n_bottom + n_server + n_top:>10,}")
        print("=" * 60)
        print("  Labels stay on client — server never sees ground-truth.")
        print("=" * 60 + "\n")

        for epoch in range(epochs):
            t0 = time.time()

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc   = self.evaluate()
            epoch_time = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["epoch_time"].append(epoch_time)

            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc * 100:5.2f}% | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Test Acc: {test_acc * 100:5.2f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

        print("\n" + "=" * 60)
        print("U-Shaped Training Complete!")
        print(f"Best Test Accuracy: {max(self.history['test_acc']) * 100:.2f}%")
        print("=" * 60)

        return self.history

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def get_smashed_data(self, images: torch.Tensor) -> torch.Tensor:
        """Return smashed data (bottom layer output) for a batch of images."""
        self.client_model.eval()
        with torch.no_grad():
            return self.client_model.forward_bottom(images.to(self.device))

    def get_server_output(self, images: torch.Tensor) -> torch.Tensor:
        """Return server output (middle layer output) for a batch of images."""
        self.client_model.eval()
        self.server_model.eval()
        with torch.no_grad():
            smashed = self.client_model.forward_bottom(images.to(self.device))
            return self.server_model(smashed)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "client_model": self.client_model.state_dict(),
                "server_model": self.server_model.state_dict(),
                "bottom_optimizer": self.bottom_optimizer.state_dict(),
                "server_optimizer": self.server_optimizer.state_dict(),
                "top_optimizer": self.top_optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.client_model.load_state_dict(ckpt["client_model"])
        self.server_model.load_state_dict(ckpt["server_model"])
        self.bottom_optimizer.load_state_dict(ckpt["bottom_optimizer"])
        self.server_optimizer.load_state_dict(ckpt["server_optimizer"])
        self.top_optimizer.load_state_dict(ckpt["top_optimizer"])
        self.history = ckpt["history"]
        print(f"Checkpoint loaded from {path}")