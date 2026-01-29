"""
Data loaders for split learning experiments.

Provides unified access to common datasets: MNIST, CIFAR-10, CIFAR-100.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os


def get_cifar10(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with standard augmentation.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Normalization values for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_mnist(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset.
    
    Note: MNIST is grayscale (1 channel). Models need to account for this.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar100(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-100 dataset with standard augmentation.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Same normalization as CIFAR-10
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_dataloader(
    dataset_name: str,
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders by dataset name.
    
    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'mnist'
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    loaders = {
        'cifar10': get_cifar10,
        'cifar100': get_cifar100,
        'mnist': get_mnist,
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    
    return loaders[dataset_name.lower()](batch_size, data_dir, num_workers)
