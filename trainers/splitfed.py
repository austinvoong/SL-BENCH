"""
SplitFed Learning Trainer

SplitFed combines split learning and federated learning:

    Split Learning:   Each client trains with a shared server model using the
                      standard smashed-data protocol. The server handles upper
                      layers; clients keep bottom layers private.

    Federated Learning: After all clients have contributed to a training round,
                        FedAvg is applied to the client-side models so that
                        every client benefits from what every other client learned
                        without sharing raw data.

Reference: Thapa et al., "SplitFed: When Federated Learning Meets Split Learning"
           https://arxiv.org/abs/2004.12088

Architecture overview:

    Client 0 ──► [Client Model 0] ──►┐
    Client 1 ──► [Client Model 1] ──►├──► [Shared Server Model] ──► Loss ──► Backprop
    Client N ──► [Client Model N] ──►┘
                                                       │
                              FedAvg after round       │
    Client 0 ◄── [Averaged Client Model] ◄─────────────┘
    Client 1 ◄── [Averaged Client Model]
    Client N ◄── [Averaged Client Model]

Training round:
    for each client in round:
        1. Client forward → smashed_data
        2. Server forward → loss → server backward
        3. Grad returned → client backward
    FedAvg(client_models)  ← synchronize client weights after each round

Notes on fidelity:
    - Clients train SEQUENTIALLY within a round (not parallel) for simplicity.
      In a real system they would run in parallel, but the FedAvg outcome is
      the same given an identical random seed.
    - The server model is updated after EVERY client batch (not averaged).
      This matches the original SplitFed-V1 paper.
    - Clients are assigned data via DataLoader partition; if a single loader
      is provided it is shared (i.e. IID setting).
"""

import copy
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# FedAvg utility
# ──────────────────────────────────────────────────────────────────────────────

def federated_average(models: List[nn.Module], weights: Optional[List[float]] = None) -> OrderedDict:
    """
    Compute the weighted average of model state_dicts (FedAvg).

    Args:
        models:  List of models to average
        weights: Optional per-client weights (e.g. number of local samples).
                 If None, uniform averaging is used.

    Returns:
        Averaged state_dict ready to load into any of the models.
    """
    if len(models) == 0:
        raise ValueError("No models provided for averaging.")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    avg_state: OrderedDict = OrderedDict()
    ref_state = models[0].state_dict()

    for key in ref_state:
        # Use float() to handle integer buffers (e.g. running_mean in BatchNorm)
        avg_state[key] = sum(
            w * m.state_dict()[key].float() for w, m in zip(weights, models)
        ).to(ref_state[key].dtype)

    return avg_state


# ──────────────────────────────────────────────────────────────────────────────
# Per-client state container
# ──────────────────────────────────────────────────────────────────────────────

class ClientState:
    """
    Encapsulates everything that belongs to one federated client:
        - Its private client model
        - Its private optimizer
        - Its local DataLoader
        - Its local training metrics
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        loader: DataLoader,
        lr: float,
        device: torch.device,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.loader = loader
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Per-client metric accumulators (reset each round)
        self.round_loss: float = 0.0
        self.round_correct: int = 0
        self.round_samples: int = 0

    def reset_metrics(self):
        self.round_loss = 0.0
        self.round_correct = 0
        self.round_samples = 0

    @property
    def round_acc(self) -> float:
        return self.round_correct / max(self.round_samples, 1)

    @property
    def avg_round_loss(self) -> float:
        return self.round_loss / max(self.round_samples, 1)


# ──────────────────────────────────────────────────────────────────────────────
# SplitFed Trainer
# ──────────────────────────────────────────────────────────────────────────────

class SplitFedTrainer:
    """
    Trainer for SplitFed learning with N clients and one shared server.

    Supports:
        - IID data (single shared DataLoader split across clients)
        - Non-IID data (one DataLoader per client)
        - Weighted FedAvg based on local dataset sizes
        - Configurable number of local steps per client per round
    """

    def __init__(
        self,
        client_model_fn,
        server_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_clients: int = 3,
        client_loaders: Optional[List[DataLoader]] = None,
        lr: float = 0.001,
        local_steps: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            client_model_fn:  Callable (no args) that returns a freshly
                              instantiated client-side model. Called once per
                              client to ensure independent weight initialization.
            server_model:     The shared server-side model (upper layers).
            train_loader:     Full training DataLoader. Used for IID partition
                              when client_loaders is None.
            test_loader:      Test DataLoader for global evaluation.
            num_clients:      Number of federated clients.
            client_loaders:   Optional list of per-client DataLoaders for non-IID
                              simulation. If None, train_loader is partitioned IID.
            lr:               Learning rate for all optimizers.
            local_steps:      Number of batches each client processes per round.
                              If None, each client processes its full local dataset.
            device:           Compute device (auto-detected if None).

        Example (IID, 4 clients):
            trainer = SplitFedTrainer(
                client_model_fn=lambda: create_split_models()[0],
                server_model=server,
                train_loader=train_loader,
                test_loader=test_loader,
                num_clients=4,
            )
            trainer.train(rounds=20)
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

        # ── Server ────────────────────────────────────────────────────────────
        self.server_model = server_model.to(self.device)
        self.server_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=lr
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── Data partitioning ─────────────────────────────────────────────────
        if client_loaders is not None:
            if len(client_loaders) != num_clients:
                raise ValueError(
                    f"len(client_loaders)={len(client_loaders)} "
                    f"does not match num_clients={num_clients}"
                )
            loaders = client_loaders
        else:
            loaders = self._partition_iid(train_loader, num_clients)

        # ── Clients ───────────────────────────────────────────────────────────
        self.clients: List[ClientState] = [
            ClientState(
                client_id=i,
                model=client_model_fn(),
                loader=loaders[i],
                lr=lr,
                device=self.device,
            )
            for i in range(num_clients)
        ]
        self.num_clients = num_clients
        self.local_steps = local_steps

        # ── Evaluation ────────────────────────────────────────────────────────
        self.test_loader = test_loader

        # ── History ───────────────────────────────────────────────────────────
        self.history: Dict[str, List] = {
            "round_train_loss": [],   # Global avg across clients
            "round_train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "client_losses": [],      # Per-client loss list each round
            "client_accs": [],
            "round_time": [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # IID data partitioning
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _partition_iid(
        loader: DataLoader, num_clients: int
    ) -> List[DataLoader]:
        """
        Partition a DataLoader's dataset into num_clients roughly equal IID splits.

        Args:
            loader:      Source DataLoader
            num_clients: Number of clients

        Returns:
            List of DataLoader, one per client
        """
        dataset = loader.dataset
        n = len(dataset)
        sizes = [n // num_clients] * num_clients
        sizes[-1] += n - sum(sizes)  # remainder goes to last client

        subsets = random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))

        return [
            DataLoader(
                subset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=getattr(loader, "num_workers", 2),
                pin_memory=True,
            )
            for subset in subsets
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Core per-client training step
    # ──────────────────────────────────────────────────────────────────────────

    def _client_train_step(
        self,
        client: ClientState,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        One split learning step for a single client.

        Mirrors VanillaSplitTrainer.train_step exactly but uses per-client
        optimizers and the shared server model/optimizer.

        Args:
            client: ClientState for this client
            images: Batch of input images
            labels: Batch of labels

        Returns:
            (loss_value, accuracy)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        # ── Client forward ────────────────────────────────────────────────────
        client.optimizer.zero_grad()
        smashed_data = client.model(images)

        # Simulate network boundary
        smashed_data_server = smashed_data.detach().requires_grad_(True)

        # ── Server forward + backward ─────────────────────────────────────────
        self.server_optimizer.zero_grad()
        outputs = self.server_model(smashed_data_server)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grad_to_client = smashed_data_server.grad.clone()
        self.server_optimizer.step()

        # ── Client backward ───────────────────────────────────────────────────
        smashed_data.backward(grad_to_client)
        client.optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────────
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).sum().item() / labels.size(0)

        return loss.item(), accuracy

    # ──────────────────────────────────────────────────────────────────────────
    # Round-level training
    # ──────────────────────────────────────────────────────────────────────────

    def _train_round(self, round_num: int) -> Tuple[float, float, List[float], List[float]]:
        """
        Execute one full training round: all clients train, then FedAvg.

        Args:
            round_num: Current round index (for progress display)

        Returns:
            (global_loss, global_acc, per_client_losses, per_client_accs)
        """
        self.server_model.train()
        for client in self.clients:
            client.model.train()
            client.reset_metrics()

        # ── Each client trains ────────────────────────────────────────────────
        for client in self.clients:
            loader_iter = iter(client.loader)
            steps = self.local_steps or len(client.loader)

            pbar = tqdm(
                range(steps),
                desc=f"  Round {round_num} | Client {client.client_id}",
                leave=False,
            )
            for step_idx in pbar:
                try:
                    images, labels = next(loader_iter)
                except StopIteration:
                    # Restart iterator if local_steps exceeds loader length
                    loader_iter = iter(client.loader)
                    images, labels = next(loader_iter)

                loss, acc = self._client_train_step(client, images, labels)

                batch_size = labels.size(0)
                client.round_loss    += loss * batch_size
                client.round_correct += int(acc * batch_size)
                client.round_samples += batch_size

                pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})

        # ── FedAvg across clients ─────────────────────────────────────────────
        sample_counts = [c.round_samples for c in self.clients]
        avg_state = federated_average(
            [c.model for c in self.clients], weights=sample_counts
        )
        for client in self.clients:
            client.model.load_state_dict(avg_state)

        # ── Aggregate round metrics ───────────────────────────────────────────
        client_losses = [c.avg_round_loss for c in self.clients]
        client_accs   = [c.round_acc      for c in self.clients]

        # Weighted global average
        total_samples = sum(c.round_samples for c in self.clients)
        global_loss = sum(
            c.avg_round_loss * c.round_samples for c in self.clients
        ) / total_samples
        global_acc = sum(
            c.round_acc * c.round_samples for c in self.clients
        ) / total_samples

        return global_loss, global_acc, client_losses, client_accs

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate using client 0's model + the shared server model.

        After FedAvg all clients share the same weights, so we only need one.

        Returns:
            (avg_loss, avg_accuracy)
        """
        self.clients[0].model.eval()
        self.server_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            smashed = self.clients[0].model(images)
            outputs = self.server_model(smashed)

            loss = self.criterion(outputs, labels)

            total_loss    += loss.item() * labels.size(0)
            _, predicted   = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    # ──────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, rounds: int, verbose: bool = True) -> Dict[str, List]:
        """
        Run SplitFed training for the specified number of rounds.

        Each round consists of:
            1. All clients train with the shared server (sequential)
            2. FedAvg applied to client models
            3. Global evaluation on test set

        Args:
            rounds:  Number of communication rounds
            verbose: Print per-round metrics

        Returns:
            History dictionary with losses, accuracies, and per-client breakdowns
        """
        n_client = sum(p.numel() for p in self.clients[0].model.parameters())
        n_server = sum(p.numel() for p in self.server_model.parameters())

        print("\nStarting SplitFed Training")
        print("=" * 60)
        print(f"  Clients                  : {self.num_clients}")
        print(f"  Client model parameters  : {n_client:>10,}  (each)")
        print(f"  Server model parameters  : {n_server:>10,}  (shared)")
        print(f"  Local steps / client     : {self.local_steps or 'full dataset'}")
        print(f"  FedAvg                   : weighted by sample count")
        print("=" * 60 + "\n")

        for rnd in range(1, rounds + 1):
            t0 = time.time()

            train_loss, train_acc, c_losses, c_accs = self._train_round(rnd)
            test_loss, test_acc = self.evaluate()
            round_time = time.time() - t0

            # Record
            self.history["round_train_loss"].append(train_loss)
            self.history["round_train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["client_losses"].append(c_losses)
            self.history["client_accs"].append(c_accs)
            self.history["round_time"].append(round_time)

            if verbose:
                client_acc_str = " | ".join(
                    f"C{i}: {a * 100:.1f}%" for i, a in enumerate(c_accs)
                )
                print(
                    f"Round {rnd:3d}/{rounds} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc * 100:5.2f}% | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Test Acc: {test_acc * 100:5.2f}% | "
                    f"Time: {round_time:.1f}s"
                )
                print(f"  └─ Per-client acc: [{client_acc_str}]")

        print("\n" + "=" * 60)
        print("SplitFed Training Complete!")
        print(f"Best Test Accuracy: {max(self.history['test_acc']) * 100:.2f}%")
        print("=" * 60)

        return self.history

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def get_client_divergence(self) -> Dict[str, float]:
        """
        Compute L2 parameter divergence between each client and the mean.

        Useful for monitoring how much clients drift before FedAvg.
        Only meaningful if called BEFORE FedAvg (i.e., mid-round).

        Returns:
            Dict mapping 'client_i' → divergence scalar
        """
        avg_state = federated_average([c.model for c in self.clients])
        divergences = {}
        for client in self.clients:
            diff = sum(
                (client.model.state_dict()[k].float() - avg_state[k].float()).norm().item() ** 2
                for k in avg_state
            ) ** 0.5
            divergences[f"client_{client.client_id}"] = diff
        return divergences

    def save_checkpoint(self, path: str):
        """Save all models and training state."""
        torch.save(
            {
                "server_model": self.server_model.state_dict(),
                "server_optimizer": self.server_optimizer.state_dict(),
                "client_models": [c.model.state_dict() for c in self.clients],
                "client_optimizers": [c.optimizer.state_dict() for c in self.clients],
                "history": self.history,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load all models and training state."""
        ckpt = torch.load(path, map_location=self.device)
        self.server_model.load_state_dict(ckpt["server_model"])
        self.server_optimizer.load_state_dict(ckpt["server_optimizer"])
        for i, client in enumerate(self.clients):
            client.model.load_state_dict(ckpt["client_models"][i])
            client.optimizer.load_state_dict(ckpt["client_optimizers"][i])
        self.history = ckpt["history"]
        print(f"Checkpoint loaded from {path}")