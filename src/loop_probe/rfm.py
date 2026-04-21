"""Minimal Laplace RFM implementation for prompt-profile detector runs.

This module is adapted from the MIT-licensed `recursive_feature_machines`
reference implementation:
https://github.com/aradha/recursive_feature_machines
commit `58bd5404bd252d1f4e7e20e01f103a3fcb5a65b2`

The in-repo copy keeps only the pieces needed for the current prompt-profile
binary detector stage:
- Laplace kernel only
- direct linear solve only
- optional diagonal metric
- explicit batching over AGOP samples / centers
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def euclidean_distances_M(
    samples: torch.Tensor,
    centers: torch.Tensor,
    M: torch.Tensor | None,
    *,
    squared: bool,
) -> torch.Tensor:
    if M is None:
        samples_norm2 = samples.pow(2).sum(-1)
        if samples.data_ptr() == centers.data_ptr():
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = centers.pow(2).sum(-1)
        distances = -2.0 * (samples @ centers.transpose(0, 1))
    elif M.ndim == 1:
        samples_norm2 = ((samples * M) * samples).sum(-1)
        if samples.data_ptr() == centers.data_ptr():
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = ((centers * M) * centers).sum(-1)
        distances = -2.0 * ((samples * M) @ centers.transpose(0, 1))
    else:
        samples_norm2 = ((samples @ M) * samples).sum(-1)
        if samples.data_ptr() == centers.data_ptr():
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = ((centers @ M) * centers).sum(-1)
        distances = -2.0 * ((samples @ M) @ centers.transpose(0, 1))
    distances = distances + samples_norm2.view(-1, 1) + centers_norm2.view(1, -1)
    if not squared:
        distances = distances.clamp_min(0.0).sqrt()
    return distances


def laplacian_M(
    samples: torch.Tensor,
    centers: torch.Tensor,
    M: torch.Tensor | None,
    bandwidth: float,
) -> torch.Tensor:
    if bandwidth <= 0.0:
        raise SystemExit("--bandwidth must be > 0.")
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat = kernel_mat.clamp_min(0.0)
    kernel_mat.mul_(-1.0 / float(bandwidth))
    kernel_mat.exp_()
    return kernel_mat


@dataclass(frozen=True)
class LaplaceRFMConfig:
    bandwidth: float
    reg: float = 1e-3
    diag: bool = False
    centering: bool = False
    sample_batch_size: int | None = None
    center_batch_size: int | None = None
    max_agop_samples: int | None = None
    device: str | torch.device | None = None
    solver: str = "cholesky"


class LaplaceRFM:
    def __init__(self, config: LaplaceRFMConfig):
        self.config = config
        self.device = _resolve_device(config.device)
        self.bandwidth = float(config.bandwidth)
        self.reg = float(config.reg)
        self.diag = bool(config.diag)
        self.centering = bool(config.centering)
        self.sample_batch_size = config.sample_batch_size
        self.center_batch_size = config.center_batch_size
        self.max_agop_samples = config.max_agop_samples
        self.solver = config.solver
        self.M: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.centers: torch.Tensor | None = None

    def _ensure_metric(self, dim: int, *, dtype: torch.dtype) -> None:
        if self.M is not None:
            return
        if self.diag:
            self.M = torch.ones(dim, device=self.device, dtype=dtype)
        else:
            self.M = torch.eye(dim, device=self.device, dtype=dtype)

    def kernel(self, samples: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        return laplacian_M(samples, centers, self.M, self.bandwidth)

    def fit_predictor(self, centers: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if centers.ndim != 2:
            raise SystemExit(f"LaplaceRFM expects 2D centers, got {tuple(centers.shape)}")
        targets = targets.reshape(-1, 1)
        if len(centers) != len(targets):
            raise SystemExit("centers and targets must have matching lengths.")

        centers = centers.to(self.device, dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)
        self._ensure_metric(centers.shape[-1], dtype=centers.dtype)
        self.centers = centers

        kernel_matrix = self.kernel(centers, centers)
        if self.reg > 0.0:
            kernel_matrix.diagonal().add_(self.reg)

        if self.solver == "solve":
            weights = torch.linalg.solve(kernel_matrix, targets)
        elif self.solver == "cholesky":
            chol = torch.linalg.cholesky(kernel_matrix)
            weights = torch.cholesky_solve(targets, chol)
        else:
            raise SystemExit(f"Unsupported LaplaceRFM solver '{self.solver}'.")

        self.weights = weights
        return weights

    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        if self.centers is None or self.weights is None:
            raise SystemExit("fit_predictor must run before predict.")
        samples = samples.to(self.device, dtype=torch.float32)
        predictions = self.kernel(samples, self.centers) @ self.weights
        return predictions.to(dtype=torch.float32).reshape(-1)

    def _update_metric_batch(self, samples: torch.Tensor) -> torch.Tensor:
        if self.centers is None or self.weights is None or self.M is None:
            raise SystemExit("fit_predictor must run before fit_M.")

        samples = samples.to(self.device, dtype=torch.float32)
        centers = self.centers
        weights = self.weights
        metric = self.M

        K = self.kernel(samples, centers)
        distances = euclidean_distances_M(samples, centers, metric, squared=False)
        safe = distances > 1e-10
        K = torch.where(safe, K / distances.clamp_min(1e-10), torch.zeros_like(K))

        n = samples.shape[0]
        p = centers.shape[0]
        d = centers.shape[1]
        c = weights.shape[1]
        batch_size = self.center_batch_size or p
        temp = torch.zeros((n, c * d), device=self.device, dtype=samples.dtype)
        if self.diag:
            metric_centers = centers * metric.view(1, -1)
            metric_samples = samples * metric.view(1, -1)
        else:
            metric_centers = centers @ metric
            metric_samples = samples @ metric

        for start in range(0, p, batch_size):
            stop = min(start + batch_size, p)
            rhs = (
                weights[start:stop].view(stop - start, c, 1)
                * metric_centers[start:stop].view(stop - start, 1, d)
            ).reshape(stop - start, c * d)
            temp.add_(K[:, start:stop] @ rhs)

        centers_term = temp.view(n, c, d)
        samples_term = (K @ weights).view(n, c, 1) * metric_samples.view(n, 1, d)
        grads = (centers_term - samples_term) / self.bandwidth
        if self.centering:
            grads = grads - grads.mean(dim=0, keepdim=True)

        if self.diag:
            return torch.einsum("ncd,ncd->d", grads, grads)
        return torch.einsum("ncd,ncD->dD", grads, grads)

    def fit_M(self, samples: torch.Tensor) -> torch.Tensor:
        if self.M is None:
            raise SystemExit("fit_predictor must run before fit_M.")
        samples = samples.to(dtype=torch.float32)
        if self.max_agop_samples is not None and len(samples) > self.max_agop_samples:
            samples = samples[: self.max_agop_samples]
        sample_batch_size = self.sample_batch_size or len(samples)
        updated = torch.zeros_like(self.M)
        for start in range(0, len(samples), sample_batch_size):
            stop = min(start + sample_batch_size, len(samples))
            updated.add_(self._update_metric_batch(samples[start:stop]))
        max_value = float(updated.max().item()) if updated.numel() else 0.0
        scale = max(max_value, 1e-30)
        self.M = updated / scale
        return self.M

    def export_state(self) -> dict[str, object]:
        if self.M is None or self.weights is None:
            raise SystemExit("Model state is empty; run fit_predictor first.")
        return {
            "bandwidth": self.bandwidth,
            "reg": self.reg,
            "diag": self.diag,
            "centering": self.centering,
            "solver": self.solver,
            "M": self.M.detach().to("cpu", dtype=torch.float32),
            "weights": self.weights.detach().to("cpu", dtype=torch.float32),
        }
