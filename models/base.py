"""
Base classes for split learning models.

In split learning, a neural network is partitioned between client and server:
- ClientModel: Holds the bottom layers, processes raw input, outputs "smashed data"
- ServerModel: Holds the top layers, receives smashed data, outputs predictions
"""

import torch
import torch.nn as nn
from typing import List


class ClientModel(nn.Module):
    """
    Client-side model in split learning.
    
    Processes raw input data and outputs intermediate activations (smashed data)
    that are sent to the server.
    """
    
    def __init__(self, layers: nn.Sequential):
        """
        Args:
            layers: Sequential container of layers for the client side
        """
        super().__init__()
        self.layers = layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through client layers.
        
        Args:
            x: Input tensor (e.g., images)
            
        Returns:
            Smashed data (intermediate activations)
        """
        return self.layers(x)


class ServerModel(nn.Module):
    """
    Server-side model in split learning.
    
    Receives smashed data from client and produces final predictions.
    """
    
    def __init__(self, layers: nn.Sequential, num_classes: int = None):
        """
        Args:
            layers: Sequential container of layers for the server side
            num_classes: Number of output classes (if applicable)
        """
        super().__init__()
        self.layers = layers
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through server layers.
        
        Args:
            x: Smashed data from client
            
        Returns:
            Model predictions
        """
        return self.layers(x)


def split_model(full_model: nn.Module, cut_layer: int) -> tuple:
    """
    Split a sequential model at a specified layer.
    
    Args:
        full_model: A model with a 'features' and 'classifier' attribute,
                   or a simple Sequential model
        cut_layer: Index of the layer where the split occurs.
                  Layers 0 to cut_layer-1 go to client,
                  Layers cut_layer onwards go to server.
    
    Returns:
        Tuple of (ClientModel, ServerModel)
    """
    # Handle models with features/classifier structure
    if hasattr(full_model, 'features') and hasattr(full_model, 'classifier'):
        all_layers = list(full_model.features) + [nn.Flatten()] + list(full_model.classifier)
    elif isinstance(full_model, nn.Sequential):
        all_layers = list(full_model)
    else:
        raise ValueError("Model must be Sequential or have 'features' and 'classifier' attributes")
    
    if cut_layer <= 0 or cut_layer >= len(all_layers):
        raise ValueError(f"cut_layer must be between 1 and {len(all_layers)-1}")
    
    client_layers = nn.Sequential(*all_layers[:cut_layer])
    server_layers = nn.Sequential(*all_layers[cut_layer:])
    
    return ClientModel(client_layers), ServerModel(server_layers)
