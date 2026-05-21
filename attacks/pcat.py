"""
attacks/pcat.py

Pseudo-Client Attack (PCAT) for split learning.

Reference:
    Gao and Zhang, "PCAT: Functionality and Data Stealing from Split Learning
    by Pseudo-Client Attack" (USENIX Security 2023)

Threat model:
    Semi-honest server. The server follows split learning but uses its current
    server-side model and a small labeled auxiliary dataset to train a pseudo
    client. Once the pseudo client can feed the server model by itself, the
    attacker trains an inverse mapping from pseudo-client features back to
    images, then applies that mapping to intercepted victim smashed data.

Implementation notes for SL-BENCH:
    - The pseudo-client architecture is VGG-like and shape-matched to the cut
      layer, because the real client architecture is unknown to the server.
    - During vanilla SL, labels are visible server-side, so update_pseudo_client
      samples auxiliary images with the same labels as the private batch.
    - A feature-moment alignment term is optional. It uses legitimately observed
      smashed data and improves stability without requiring protocol deviation.
"""

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.fora import SubstituteClient
from attacks.inverse_network import InverseNetwork
from metrics.reconstruction import (
    compute_psnr,
    compute_ssim,
    distance_correlation,
)


class PseudoClient(SubstituteClient):
    """Shape-matched pseudo-client used by PCAT."""

    @staticmethod
    def for_smashed_shape(
        smashed_channels: int,
        smashed_spatial: int,
        img_channels: int = 3,
        img_spatial: int = 32,
    ) -> "PseudoClient":
        return PseudoClient(
            in_channels=img_channels,
            out_channels=smashed_channels,
            out_spatial=smashed_spatial,
            in_spatial=img_spatial,
        )


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def feature_moment_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match first and second moments of two smashed-data batches.

    PCAT's main objective is task loss through the server model. This optional
    term uses observed smashed data to keep pseudo-client features on a similar
    scale to victim features, which makes inverse-map training less brittle.
    """
    source = source.float()
    target = target.float()

    if source.dim() > 2:
        reduce_dims = (0, *range(2, source.dim()))
    else:
        reduce_dims = (0,)

    src_mean = source.mean(dim=reduce_dims)
    tgt_mean = target.mean(dim=reduce_dims)
    src_std = source.std(dim=reduce_dims, unbiased=False)
    tgt_std = target.std(dim=reduce_dims, unbiased=False)

    return F.mse_loss(src_mean, tgt_mean) + F.mse_loss(src_std, tgt_std)


class PCATAttack:
    """
    Pseudo-Client Attack.

    Typical use inside a vanilla split-learning loop:

        pcat = PCATAttack(..., aux_loader=aux_loader)
        for images, labels in train_loader:
            smashed = client_model(images)
            pcat.update_pseudo_client(
                server_model=server_model,
                private_labels=labels,
                smashed_priv=smashed.detach(),
            )
            ... normal split-learning server/client update ...

        pcat.train_inverse_network(epochs=30)
        recon = pcat.reconstruct_batch(client_model(images))

    The post-hoc fit_pseudo_client() method is also provided for dashboard runs
    that cannot hook into each SL iteration, but the in-loop workflow is closer
    to the paper.
    """

    def __init__(
        self,
        smashed_channels: int,
        smashed_spatial: int,
        aux_loader: DataLoader,
        cut_layer: int = 2,
        img_channels: int = 3,
        img_spatial: int = 32,
        lr_pseudo: float = 1e-3,
        lr_inverse: float = 1e-3,
        lambda_moment: float = 0.05,
        cache_auxiliary: bool = True,
        max_aux_cache: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device is not None else _auto_device()
        print(f"[PCAT] Using device: {self.device}")

        self.smashed_channels = smashed_channels
        self.smashed_spatial = smashed_spatial
        self.cut_layer = cut_layer
        self.lambda_moment = lambda_moment
        self.aux_loader = aux_loader
        self._aux_iter: Optional[Iterator] = None

        self.pseudo_client = PseudoClient.for_smashed_shape(
            smashed_channels=smashed_channels,
            smashed_spatial=smashed_spatial,
            img_channels=img_channels,
            img_spatial=img_spatial,
        ).to(self.device)

        self.inverse_net = InverseNetwork.for_cut_layer(cut_layer).to(self.device)

        self.opt_pseudo = torch.optim.Adam(self.pseudo_client.parameters(), lr=lr_pseudo)
        self.opt_inverse = torch.optim.Adam(self.inverse_net.parameters(), lr=lr_inverse)

        self._aux_images: Optional[torch.Tensor] = None
        self._aux_labels: Optional[torch.Tensor] = None
        self._label_to_indices: Dict[int, torch.Tensor] = {}
        if cache_auxiliary:
            self._build_aux_cache(max_samples=max_aux_cache)

        self._snapshot_list: List[torch.Tensor] = []
        self.history: Dict[str, List[float]] = {
            "pseudo_loss": [],
            "pseudo_ce": [],
            "pseudo_moment": [],
            "pseudo_acc": [],
            "inverse_loss": [],
            "functionality_acc": [],
            "val_ssim": [],
            "val_psnr": [],
        }

    def _build_aux_cache(self, max_samples: Optional[int] = None) -> None:
        """Cache auxiliary data on CPU so label-matched sampling is cheap."""
        images_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        collected = 0

        for batch in self.aux_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                labels = None

            take = images.size(0)
            if max_samples is not None:
                take = min(take, max_samples - collected)
            if take <= 0:
                break

            images_list.append(images[:take].detach().cpu())
            if labels is not None:
                labels_list.append(labels[:take].detach().cpu().long())

            collected += take
            if max_samples is not None and collected >= max_samples:
                break

        if not images_list:
            return

        self._aux_images = torch.cat(images_list, dim=0)
        if labels_list:
            self._aux_labels = torch.cat(labels_list, dim=0)
            for label in self._aux_labels.unique().tolist():
                label_int = int(label)
                self._label_to_indices[label_int] = torch.nonzero(
                    self._aux_labels == label_int,
                    as_tuple=False,
                ).view(-1)

        label_state = "with labels" if self._aux_labels is not None else "without labels"
        print(f"[PCAT] Cached {len(self._aux_images):,} auxiliary samples {label_state}.")

    def _next_aux_batch(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the next auxiliary batch, cycling forever."""
        if self._aux_iter is None:
            self._aux_iter = iter(self.aux_loader)

        try:
            batch = next(self._aux_iter)
        except StopIteration:
            self._aux_iter = iter(self.aux_loader)
            batch = next(self._aux_iter)

        if isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None

        images = images.to(self.device)
        labels = labels.to(self.device).long() if labels is not None else None
        return images, labels

    def _resize_batch(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if images.size(0) == batch_size:
            return images, labels

        if images.size(0) > batch_size:
            labels = labels[:batch_size] if labels is not None else None
            return images[:batch_size], labels

        repeats = (batch_size // images.size(0)) + 1
        images = images.repeat(repeats, 1, 1, 1)[:batch_size]
        if labels is not None:
            labels = labels.repeat(repeats)[:batch_size]
        return images, labels

    def _sample_label_matched(
        self,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one auxiliary image per target label.

        If a label is missing from the auxiliary cache, fall back to a random
        auxiliary sample and use that sample's real label for the task loss.
        """
        if self._aux_images is None or self._aux_labels is None:
            images, aux_labels = self._next_aux_batch()
            if aux_labels is None:
                raise ValueError("PCAT needs labels in aux_loader for pseudo-client training.")
            return self._resize_batch(images, aux_labels, labels.size(0))

        selected: List[int] = []
        labels_cpu = labels.detach().cpu().long()
        n_aux = len(self._aux_images)

        for label in labels_cpu.tolist():
            pool = self._label_to_indices.get(int(label))
            if pool is None or len(pool) == 0:
                idx = int(torch.randint(n_aux, (1,)).item())
            else:
                idx = int(pool[torch.randint(len(pool), (1,)).item()].item())
            selected.append(idx)

        idx_tensor = torch.tensor(selected, dtype=torch.long)
        xaux = self._aux_images.index_select(0, idx_tensor).to(self.device)
        yaux = self._aux_labels.index_select(0, idx_tensor).to(self.device).long()
        return xaux, yaux

    def _train_pseudo_on_batch(
        self,
        server_model: nn.Module,
        xaux: torch.Tensor,
        yaux: torch.Tensor,
        smashed_priv: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One pseudo-client optimization step."""
        server_model = server_model.to(self.device)
        xaux = xaux.to(self.device)
        yaux = yaux.to(self.device).long()
        smashed_priv = smashed_priv.to(self.device).detach() if smashed_priv is not None else None

        server_was_training = server_model.training
        previous_requires_grad = [p.requires_grad for p in server_model.parameters()]

        server_model.eval()
        for p in server_model.parameters():
            p.requires_grad_(False)

        self.pseudo_client.train()
        self.opt_pseudo.zero_grad()

        zaux = self.pseudo_client(xaux)
        logits = server_model(zaux)
        loss_ce = F.cross_entropy(logits, yaux)

        if smashed_priv is not None and self.lambda_moment > 0:
            if smashed_priv.size(0) != zaux.size(0):
                smashed_priv = smashed_priv[: zaux.size(0)]
            loss_moment = feature_moment_loss(zaux, smashed_priv)
        else:
            loss_moment = torch.zeros((), device=self.device)

        loss = loss_ce + self.lambda_moment * loss_moment
        loss.backward()
        self.opt_pseudo.step()

        for param, requires_grad in zip(server_model.parameters(), previous_requires_grad):
            param.requires_grad_(requires_grad)
        if server_was_training:
            server_model.train()

        with torch.no_grad():
            pseudo_acc = logits.argmax(dim=1).eq(yaux).float().mean().item()

        metrics = {
            "pseudo_loss": float(loss.item()),
            "pseudo_ce": float(loss_ce.item()),
            "pseudo_moment": float(loss_moment.item()),
            "pseudo_acc": float(pseudo_acc),
        }
        for key, value in metrics.items():
            self.history[key].append(value)
        return metrics

    def update_pseudo_client(
        self,
        server_model: nn.Module,
        private_labels: Optional[torch.Tensor] = None,
        smashed_priv: Optional[torch.Tensor] = None,
        steps: int = 1,
    ) -> Dict[str, float]:
        """
        Update the pseudo-client during normal split learning.

        Args:
            server_model: Current server-side model.
            private_labels: Labels observed by the server in vanilla SL.
            smashed_priv: Victim smashed data from the same iteration.
            steps: Number of pseudo-client optimization steps.
        """
        if private_labels is None:
            xaux, yaux = self._next_aux_batch()
            if yaux is None:
                raise ValueError("private_labels or auxiliary labels are required for PCAT.")
            if smashed_priv is not None:
                xaux, yaux = self._resize_batch(xaux, yaux, smashed_priv.size(0))
        else:
            xaux, yaux = self._sample_label_matched(private_labels)

        metric_sums = {
            "pseudo_loss": 0.0,
            "pseudo_ce": 0.0,
            "pseudo_moment": 0.0,
            "pseudo_acc": 0.0,
        }
        for _ in range(max(steps, 1)):
            metrics = self._train_pseudo_on_batch(server_model, xaux, yaux, smashed_priv)
            for key in metric_sums:
                metric_sums[key] += metrics[key]

        return {key: value / max(steps, 1) for key, value in metric_sums.items()}

    def fit_pseudo_client(
        self,
        server_model: nn.Module,
        epochs: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Post-hoc pseudo-client training using the final server model.

        This is useful when the caller cannot hook PCAT into every SL iteration.
        The in-loop update_pseudo_client workflow is still the preferred setup.
        """
        print(f"\n{'-' * 60}")
        print("  PCAT Phase 1 - Pseudo-Client Functionality Stealing")
        print(f"{'-' * 60}")
        print(f"  Epochs        : {epochs}")
        print(f"  Lambda moment : {self.lambda_moment}")
        print(f"{'-' * 60}\n")

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            total_batches = 0

            pbar = tqdm(
                self.aux_loader,
                desc=f"  [PCAT Phase 1] Epoch {epoch + 1:3d}/{epochs}",
                leave=False,
            )
            for batch in pbar:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError("PCAT fit_pseudo_client requires labeled auxiliary data.")

                xaux = batch[0].to(self.device)
                yaux = batch[1].to(self.device).long()
                metrics = self._train_pseudo_on_batch(server_model, xaux, yaux)

                total_loss += metrics["pseudo_loss"]
                total_acc += metrics["pseudo_acc"]
                total_batches += 1
                pbar.set_postfix({
                    "loss": f"{metrics['pseudo_loss']:.4f}",
                    "acc": f"{metrics['pseudo_acc'] * 100:.1f}%",
                })

            avg_loss = total_loss / max(total_batches, 1)
            avg_acc = total_acc / max(total_batches, 1)
            self.history["functionality_acc"].append(avg_acc)
            if verbose:
                print(
                    f"  [PCAT Phase 1] Epoch {epoch + 1:3d}/{epochs} | "
                    f"Pseudo Loss: {avg_loss:.4f} | Pseudo Acc: {avg_acc * 100:.2f}%"
                )

        return self.history

    def train_inverse_network(
        self,
        epochs: int = 30,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train reverse mapping f_inv(PseudoClient(Xaux)) -> Xaux."""
        for p in self.pseudo_client.parameters():
            p.requires_grad_(False)

        self.pseudo_client.eval()
        self.inverse_net.train()

        print(f"\n{'-' * 60}")
        print("  PCAT Phase 2 - Reverse Mapping Training")
        print(f"{'-' * 60}")
        print(f"  Epochs : {epochs}")
        print(f"{'-' * 60}\n")

        for epoch in range(epochs):
            total_loss = 0.0
            total_samples = 0

            pbar = tqdm(
                self.aux_loader,
                desc=f"  [PCAT Phase 2] Epoch {epoch + 1:3d}/{epochs}",
                leave=False,
            )
            for batch in pbar:
                xaux = batch[0] if isinstance(batch, (list, tuple)) else batch
                xaux = xaux.to(self.device)

                self.opt_inverse.zero_grad()
                with torch.no_grad():
                    zaux = self.pseudo_client(xaux)
                recon = self.inverse_net(zaux)
                loss = F.mse_loss(recon, xaux)

                loss.backward()
                self.opt_inverse.step()

                total_loss += loss.item() * xaux.size(0)
                total_samples += xaux.size(0)
                pbar.set_postfix({"mse": f"{loss.item():.5f}"})

            avg_loss = total_loss / max(total_samples, 1)
            self.history["inverse_loss"].append(avg_loss)
            if verbose:
                print(
                    f"  [PCAT Phase 2] Epoch {epoch + 1:3d}/{epochs} | "
                    f"MSE Loss: {avg_loss:.5f}"
                )

        for p in self.pseudo_client.parameters():
            p.requires_grad_(True)

        return self.history

    def add_to_snapshot(self, smashed_priv: torch.Tensor) -> None:
        self._snapshot_list.append(smashed_priv.detach().cpu())

    def clear_snapshot(self) -> None:
        self._snapshot_list.clear()

    @torch.no_grad()
    def reconstruct_from_snapshot(self, batch_size: int = 128) -> torch.Tensor:
        if not self._snapshot_list:
            raise RuntimeError("Snapshot buffer is empty. Call add_to_snapshot() first.")

        self.inverse_net.eval()
        zsnap = torch.cat(self._snapshot_list, dim=0)
        recon_batches = []
        for start in range(0, len(zsnap), batch_size):
            zbatch = zsnap[start : start + batch_size].to(self.device)
            recon_batches.append(self.inverse_net(zbatch).cpu())
        return torch.cat(recon_batches, dim=0)

    @torch.no_grad()
    def reconstruct_batch(self, smashed: torch.Tensor) -> torch.Tensor:
        self.inverse_net.eval()
        return self.inverse_net(smashed.to(self.device))

    @torch.no_grad()
    def measure_functionality(
        self,
        server_model: nn.Module,
        data_loader: DataLoader,
        n_batches: int = 20,
    ) -> Dict[str, float]:
        """
        Measure stolen functionality: server_model(pseudo_client(x)) accuracy.
        """
        server_model = server_model.to(self.device).eval()
        self.pseudo_client.eval()

        correct = 0
        total = 0
        losses = []

        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device).long()
            logits = server_model(self.pseudo_client(images))
            losses.append(F.cross_entropy(logits, labels).item())
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        loss = sum(losses) / max(len(losses), 1)
        self.history["functionality_acc"].append(acc)
        return {"functionality_acc": acc, "functionality_loss": loss}

    @torch.no_grad()
    def evaluate(
        self,
        original_images: torch.Tensor,
        reconstructed: torch.Tensor,
        smashed_data: Optional[torch.Tensor] = None,
        label: str = "PCAT",
    ) -> Dict[str, float]:
        original_images = original_images.to(self.device)
        reconstructed = reconstructed.to(self.device).clamp(-1, 1)

        results = {
            "ssim": compute_ssim(reconstructed, original_images).item(),
            "psnr": compute_psnr(reconstructed, original_images).item(),
        }
        if smashed_data is not None:
            results["dcor"] = distance_correlation(
                original_images,
                smashed_data.to(self.device),
            ).item()

        print(f"\n{'-' * 50}")
        print(f"  {label} Reconstruction Metrics")
        print(f"{'-' * 50}")
        print(f"  SSIM : {results['ssim']:.4f}")
        print(f"  PSNR : {results['psnr']:.2f} dB")
        if "dcor" in results:
            print(f"  dCor : {results['dcor']:.4f}")
        print(f"{'-' * 50}\n")

        return results

    @torch.no_grad()
    def evaluate_on_loader(
        self,
        data_loader: DataLoader,
        client_model: nn.Module,
        n_batches: int = 20,
    ) -> Dict[str, float]:
        client_model = client_model.to(self.device).eval()
        self.inverse_net.eval()

        all_ssim = []
        all_psnr = []
        originals_for_dcor = []
        smashed_for_dcor = []

        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break
            images = images.to(self.device)
            smashed = client_model(images)
            recon = self.inverse_net(smashed).clamp(-1, 1)

            all_ssim.append(compute_ssim(recon, images).item())
            all_psnr.append(compute_psnr(recon, images).item())

            if batch_idx < 4:
                originals_for_dcor.append(images)
                smashed_for_dcor.append(smashed)

        results = {
            "ssim": sum(all_ssim) / max(len(all_ssim), 1),
            "psnr": sum(all_psnr) / max(len(all_psnr), 1),
        }
        if originals_for_dcor:
            originals = torch.cat(originals_for_dcor, dim=0)
            smashed = torch.cat(smashed_for_dcor, dim=0)
            results["dcor"] = distance_correlation(originals, smashed).item()
        else:
            results["dcor"] = float("nan")

        self.history["val_ssim"].append(results["ssim"])
        self.history["val_psnr"].append(results["psnr"])
        return results

    def save(self, path: str) -> None:
        torch.save({
            "pseudo_client": self.pseudo_client.state_dict(),
            "inverse_net": self.inverse_net.state_dict(),
            "opt_pseudo": self.opt_pseudo.state_dict(),
            "opt_inverse": self.opt_inverse.state_dict(),
            "cut_layer": self.cut_layer,
            "smashed_channels": self.smashed_channels,
            "smashed_spatial": self.smashed_spatial,
            "lambda_moment": self.lambda_moment,
            "history": self.history,
        }, path)
        print(f"[PCAT] Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.pseudo_client.load_state_dict(ckpt["pseudo_client"])
        self.inverse_net.load_state_dict(ckpt["inverse_net"])
        self.opt_pseudo.load_state_dict(ckpt["opt_pseudo"])
        self.opt_inverse.load_state_dict(ckpt["opt_inverse"])
        self.history = ckpt.get("history", self.history)
        print(f"[PCAT] Loaded checkpoint from {path}")
