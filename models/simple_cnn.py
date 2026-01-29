"""
Simple CNN architecture for CIFAR-10 with configurable split points.

This model is designed to be easily split at various layers for split learning
experiments. It achieves ~85% accuracy on CIFAR-10 with proper training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .base import ClientModel, ServerModel


class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 classification.
    
    Architecture:
        Layer 0: Conv2d(3, 32) + ReLU + MaxPool    -> 16x16x32
        Layer 1: Conv2d(32, 64) + ReLU + MaxPool   -> 8x8x64
        Layer 2: Conv2d(64, 128) + ReLU + MaxPool  -> 4x4x128
        Layer 3: Flatten + Linear(2048, 256) + ReLU
        Layer 4: Linear(256, num_classes)
    
    Possible cut points: 1, 2, 3, 4 (after respective layer)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional layers (feature extraction)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
        
        # Classifier layers
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Linear(256, num_classes)
        
        # Store layers in order for easy splitting
        self._layers = [self.conv1, self.conv2, self.conv3, 
                        self.flatten, self.fc1, self.fc2]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def get_smashed_data_size(self, cut_layer: int, input_size: Tuple[int, int] = (32, 32)) -> Tuple:
        """
        Calculate the output size at a given cut layer.
        
        Args:
            cut_layer: Layer index (1-5)
            input_size: Input image size (H, W)
            
        Returns:
            Shape of smashed data (excluding batch dimension)
        """
        sizes = {
            1: (32, 16, 16),    # After conv1
            2: (64, 8, 8),      # After conv2
            3: (128, 4, 4),     # After conv3
            4: (2048,),         # After flatten
            5: (256,),          # After fc1
        }
        return sizes.get(cut_layer, None)


def create_split_models(
    num_classes: int = 10,
    cut_layer: int = 2
) -> Tuple[ClientModel, ServerModel]:
    """
    Create client and server models by splitting SimpleCNN.
    
    Args:
        num_classes: Number of output classes
        cut_layer: Where to split the model (1-5)
                  1 = after first conv block
                  2 = after second conv block (recommended)
                  3 = after third conv block
                  4 = after flatten
                  5 = after first FC layer
    
    Returns:
        Tuple of (client_model, server_model)
    
    Example:
        >>> client, server = create_split_models(num_classes=10, cut_layer=2)
        >>> # Client has: conv1, conv2
        >>> # Server has: conv3, flatten, fc1, fc2
    """
    if cut_layer < 1 or cut_layer > 5:
        raise ValueError(f"cut_layer must be between 1 and 5, got {cut_layer}")
    
    # Build client model
    client_layers = []
    if cut_layer >= 1:
        client_layers.append(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
    if cut_layer >= 2:
        client_layers.append(nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
    if cut_layer >= 3:
        client_layers.append(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
    if cut_layer >= 4:
        client_layers.append(nn.Flatten())
        client_layers.append(nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ))
    if cut_layer >= 5:
        client_layers.append(nn.Linear(256, num_classes))
    
    # Build server model
    server_layers = []
    if cut_layer < 1:
        server_layers.append(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
    if cut_layer < 2:
        server_layers.append(nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
    if cut_layer < 3:
        server_layers.append(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
    if cut_layer < 4:
        server_layers.append(nn.Flatten())
        # Calculate input size based on cut layer
        if cut_layer == 1:
            fc_input = 64 * 8 * 8  # After conv2, conv3
        elif cut_layer == 2:
            fc_input = 128 * 4 * 4  # After conv3
        elif cut_layer == 3:
            fc_input = 128 * 4 * 4
        server_layers.append(nn.Sequential(
            nn.Linear(fc_input if cut_layer < 3 else 128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ))
    if cut_layer < 5:
        if cut_layer == 4:
            # fc1 is on client, server just has fc2
            pass
        server_layers.append(nn.Linear(256, num_classes))
    
    client_model = ClientModel(nn.Sequential(*client_layers))
    server_model = ServerModel(nn.Sequential(*server_layers), num_classes=num_classes)
    
    return client_model, server_model


# Simpler, more direct approach - recommended
def create_split_simple_cnn(cut_layer: int = 2, num_classes: int = 10):
    """
    Create split models with a cleaner approach.
    
    Cut layer options:
        1: Client=[conv1], Server=[conv2, conv3, fc1, fc2]
        2: Client=[conv1, conv2], Server=[conv3, fc1, fc2]  <- RECOMMENDED
        3: Client=[conv1, conv2, conv3], Server=[fc1, fc2]
    
    Returns:
        Tuple of (client_model, server_model)
    """
    
    if cut_layer == 1:
        client = ClientModel(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
        server = ServerModel(nn.Sequential(
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # classifier
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        ), num_classes=num_classes)
        
    elif cut_layer == 2:
        client = ClientModel(nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
        server = ServerModel(nn.Sequential(
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # classifier
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        ), num_classes=num_classes)
        
    elif cut_layer == 3:
        client = ClientModel(nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ))
        server = ServerModel(nn.Sequential(
            # classifier only
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        ), num_classes=num_classes)
    else:
        raise ValueError(f"cut_layer must be 1, 2, or 3. Got {cut_layer}")
    
    return client, server
