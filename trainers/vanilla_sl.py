"""
Vanilla Split Learning Trainer

Implements the basic split learning training protocol where:
1. Client processes input through bottom layers → produces smashed data
2. Smashed data is "sent" to server (simulated)
3. Server completes forward pass, computes loss, backprops
4. Server "sends" gradients back to client (simulated)
5. Client completes backprop through bottom layers

This is a single-client, synchronous implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Callable
from tqdm import tqdm
import time


class VanillaSplitTrainer:
    """
    Trainer for vanilla split learning.
    
    Simulates the split learning protocol without actual network communication.
    The "split" is implemented by detaching tensors at the cut point.
    """
    
    def __init__(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 0.001,
        device: str = None
    ):
        """
        Initialize the split learning trainer.
        
        Args:
            client_model: Model running on client (bottom layers)
            server_model: Model running on server (top layers)
            train_loader: Training data loader
            test_loader: Test data loader
            lr: Learning rate
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Models
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Optimizers (separate for client and server)
        self.client_optimizer = torch.optim.Adam(
            self.client_model.parameters(), lr=lr
        )
        self.server_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=lr
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': []
        }
    
    def train_step(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Perform one training step with split learning protocol.
        
        Args:
            images: Batch of input images
            labels: Batch of labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # ============ CLIENT SIDE ============
        self.client_optimizer.zero_grad()
        
        # Forward pass through client model
        smashed_data = self.client_model(images)
        
        # Prepare smashed data for "transmission" to server
        # .detach() breaks the computational graph (simulates network boundary)
        # .requires_grad_(True) allows server to compute gradients w.r.t. this tensor
        smashed_data_server = smashed_data.detach().clone().requires_grad_(True)
        
        # ============ SERVER SIDE ============
        self.server_optimizer.zero_grad()
        
        # Forward pass through server model
        outputs = self.server_model(smashed_data_server)
        
        # Compute loss
        loss = self.criterion(outputs, labels)
        
        # Backward pass on server
        loss.backward()
        
        # Get gradients to "send" back to client
        grad_to_client = smashed_data_server.grad.clone()
        
        # Update server model
        self.server_optimizer.step()
        
        # ============ CLIENT SIDE (continued) ============
        # Backward pass on client using gradients from server
        smashed_data.backward(grad_to_client)
        
        # Update client model
        self.client_optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0)
        
        return loss.item(), accuracy
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.client_model.train()
        self.server_model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            loss, acc = self.train_step(images, labels)
            
            total_loss += loss * labels.size(0)
            total_correct += acc * labels.size(0)
            total_samples += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{acc:.4f}'
            })
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.client_model.eval()
        self.server_model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass (no gradient tracking needed)
            smashed_data = self.client_model(images)
            outputs = self.server_model(smashed_data)
            
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(
        self,
        epochs: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the split learning model.
        
        Args:
            epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting Split Learning Training")
        print(f"{'='*50}")
        print(f"Client model parameters: {sum(p.numel() for p in self.client_model.parameters()):,}")
        print(f"Server model parameters: {sum(p.numel() for p in self.server_model.parameters()):,}")
        print(f"{'='*50}\n")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['epoch_time'].append(epoch_time)
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc*100:5.2f}% | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"Test Acc: {test_acc*100:5.2f}% | "
                      f"Time: {epoch_time:.1f}s")
        
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Test Accuracy: {max(self.history['test_acc'])*100:.2f}%")
        print(f"{'='*50}")
        
        return self.history
    
    def get_smashed_data(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get the smashed data (intermediate activations) for a batch of images.
        
        Useful for analyzing what information is leaked at the cut layer.
        
        Args:
            images: Batch of input images
            
        Returns:
            Smashed data tensor
        """
        self.client_model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            smashed_data = self.client_model(images)
        return smashed_data
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'client_model': self.client_model.state_dict(),
            'server_model': self.server_model.state_dict(),
            'client_optimizer': self.client_optimizer.state_dict(),
            'server_optimizer': self.server_optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.client_model.load_state_dict(checkpoint['client_model'])
        self.server_model.load_state_dict(checkpoint['server_model'])
        self.client_optimizer.load_state_dict(checkpoint['client_optimizer'])
        self.server_optimizer.load_state_dict(checkpoint['server_optimizer'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")
