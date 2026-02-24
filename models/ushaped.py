"""
U-Shaped Split Learning Model Architecture

In U-shaped split learning, the network is partitioned into THREE segments:
    [Client Bottom] → smashed_data → [Server Middle] → server_output → [Client Top] → loss

Key privacy property: Labels NEVER leave the client. The server only sees
intermediate activations and never touches raw inputs or ground-truth labels.

                    CLIENT                  SERVER                  CLIENT
    Input ──► [Bottom Layers] ──► smashed ──► [Middle Layers] ──► output ──► [Top Layers] ──► Loss
                                    (sent)                          (sent)
                                    ─────────────────────────────────────────────────────────────►
                                               Forward Pass

                                    ◄─────────────────────────────────────────────────────────────
                    grad_bottom ◄── grad_smashed ◄── grad_output ◄── loss.backward()
                                              Backward Pass

Supported split configurations for SimpleCNN:
    Config 1:  client_bottom=[conv1],        server=[conv2, conv3],  client_top=[flatten, fc1, fc2]
    Config 2:  client_bottom=[conv1, conv2], server=[conv3],         client_top=[flatten, fc1, fc2]  ← RECOMMENDED
"""

import torch
import torch.nn as nn
from typing import Tuple

from .base import ClientModel, ServerModel


class UShapedClientModel(nn.Module):
    """
    Client-side model for U-shaped split learning.

    Holds both the bottom layers (feature extraction before server)
    and top layers (classifier after server). Computes the loss itself
    since labels stay on the client.

    Usage:
        smashed = model.forward_bottom(x)          # Step 1: send to server
        logits  = model.forward_top(server_output) # Step 3: after server returns
        loss    = criterion(logits, labels)        # Step 4: loss on client
    """

    def __init__(
        self,
        bottom_layers: nn.Sequential,
        top_layers: nn.Sequential,
    ):
        """
        Args:
            bottom_layers: Layers before the server segment (input → smashed data)
            top_layers:    Layers after the server segment (server output → logits)
        """
        super().__init__()
        self.bottom = bottom_layers
        self.top = top_layers

    def forward_bottom(self, x: torch.Tensor) -> torch.Tensor:
        """
        First half of the forward pass (client → server direction).

        Args:
            x: Raw input tensor (e.g. images)

        Returns:
            Smashed data to be transmitted to server
        """
        return self.bottom(x)

    def forward_top(self, server_output: torch.Tensor) -> torch.Tensor:
        """
        Second half of the forward pass (server → client direction).

        Args:
            server_output: Activations returned from the server middle segment

        Returns:
            Final logits / predictions
        """
        return self.top(server_output)

    def forward(self, x: torch.Tensor, server_fn) -> torch.Tensor:
        """
        Complete forward pass when server is callable in the same process.

        Args:
            x:         Raw input tensor
            server_fn: Callable representing the server forward pass

        Returns:
            Final logits
        """
        smashed = self.forward_bottom(x)
        server_out = server_fn(smashed)
        return self.forward_top(server_out)

    def bottom_parameters(self):
        return self.bottom.parameters()

    def top_parameters(self):
        return self.top.parameters()


def create_ushaped_models(
    cut_1: int = 2,
    num_classes: int = 10,
) -> Tuple[UShapedClientModel, ServerModel]:
    """
    Create U-shaped split models from SimpleCNN architecture.

    The network is divided at two cut points:
        Layers [0, cut_1)        → client bottom
        Layers [cut_1, 3)        → server middle  (always ends before classifier)
        Layers [3, end]          → client top      (flatten + fc1 + fc2)

    Args:
        cut_1:       Where client hands off to server.
                     1 → client_bottom = [conv1]
                     2 → client_bottom = [conv1, conv2]  ← recommended
        num_classes: Number of output classes

    Returns:
        Tuple of (UShapedClientModel, ServerModel)

    Architecture breakdown by cut_1:
        cut_1=1: bottom=[conv1]        | server=[conv2, conv3] | top=[flatten, fc1, fc2]
        cut_1=2: bottom=[conv1, conv2] | server=[conv3]        | top=[flatten, fc1, fc2]
    """
    if cut_1 not in (1, 2):
        raise ValueError(f"cut_1 must be 1 or 2 for SimpleCNN. Got {cut_1}.")

    # ── Client Bottom ────────────────────────────────────────────────────────
    if cut_1 == 1:
        bottom = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 32×32 → 16×16
        )
        server_in_channels = 32
    else:  # cut_1 == 2
        bottom = nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 32×32 → 16×16
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 16×16 → 8×8
        )
        server_in_channels = 64

    # ── Server Middle ─────────────────────────────────────────────────────────
    # Always processes remaining conv layers, always outputs 128×4×4
    if cut_1 == 1:
        server_layers = nn.Sequential(
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 16×16 → 8×8
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 8×8 → 4×4
        )
    else:  # cut_1 == 2
        server_layers = nn.Sequential(
            # conv3 only
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 8×8 → 4×4
        )

    server_model = ServerModel(server_layers)

    # ── Client Top ────────────────────────────────────────────────────────────
    # Always receives 128×4×4 tensor from server, computes final logits
    top = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    client_model = UShapedClientModel(bottom_layers=bottom, top_layers=top)

    return client_model, server_model


def smashed_data_shape_ushaped(cut_1: int) -> Tuple[int, ...]:
    """Return the shape of smashed data (excluding batch dim) for each cut_1 value."""
    shapes = {
        1: (32, 16, 16),  # After conv1
        2: (64, 8, 8),    # After conv2
    }
    if cut_1 not in shapes:
        raise ValueError(f"cut_1 must be 1 or 2. Got {cut_1}.")
    return shapes[cut_1]