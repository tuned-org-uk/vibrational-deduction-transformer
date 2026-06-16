"""
Dataset helpers for Wiring Autoencoder.

Supported datasets:
    cora     — Cora citation network (PyG, 7 classes, 1433-dim)
    pubmed   — PubMed citation network (PyG, 3 classes, 500-dim)
    mnist    — MNIST flattened pixels (784-dim), built into synthetic mini-graph
    custom   — load CSV / npy embedding matrix
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import os
import numpy as np


class NodeEmbeddingDataset(Dataset):
    """
    Simple dataset wrapping a node embedding matrix.

    Parameters
    ----------
    E : Tensor  (N, D)  — full embedding table
    labels : Tensor  (N,) or None
    indices : Tensor  (M,) or None  — subset of nodes (e.g. train split)
    """

    def __init__(
        self,
        E: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.E = E
        self.labels = labels
        self.indices = indices if indices is not None else torch.arange(len(E))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        node_idx = self.indices[idx].item()
        out = {"x": self.E[node_idx], "node_idx": torch.tensor(node_idx)}
        if self.labels is not None:
            out["label"] = self.labels[node_idx]
        return out


def load_dataset(
    name: str,
    root: str = "./data",
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    device: str = "cpu",
) -> dict:
    """
    Load a dataset and return:
        E       ... (N, D)  embedding table (features)
        labels  ... (N,)    class labels
        splits  ... {train, val, test}  NodeEmbeddingDataset
        meta    ... dict with dataset metadata

    Returns
    -------
    dict with keys: E, labels, train_dataset, val_dataset, test_dataset, meta
    """
    os.makedirs(root, exist_ok=True)

    if name in ("cora", "pubmed"):
        return _load_pyg(name, root, split, device)
    elif name == "mnist":
        return _load_mnist(root, split, device)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'cora', 'pubmed', or 'mnist'.")


def _load_pyg(name: str, root: str, split, device: str) -> dict:
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root=root, name=name.capitalize())
    data = ds[0]
    E      = data.x.to(device)
    labels = data.y.to(device)
    N = E.shape[0]
    idx = torch.randperm(N)
    n_train = int(split[0] * N)
    n_val   = int(split[1] * N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return {
        "E": E, "labels": labels,
        "train_dataset": NodeEmbeddingDataset(E, labels, train_idx),
        "val_dataset":   NodeEmbeddingDataset(E, labels, val_idx),
        "test_dataset":  NodeEmbeddingDataset(E, labels, test_idx),
        "meta": {"name": name, "n_nodes": N, "n_classes": ds.num_classes, "feat_dim": E.shape[1]},
    }


def _load_mnist(root: str, split, device: str) -> dict:
    from torchvision.datasets import MNIST
    from torchvision import transforms
    ds_train = MNIST(root=root, train=True,  download=True, transform=transforms.ToTensor())
    ds_test  = MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    # Subsample to 5000 nodes for speed
    N = 5000
    imgs, labs = [], []
    for i in range(N):
        img, lbl = ds_train[i]
        imgs.append(img.view(-1))   # 784
        labs.append(lbl)
    E      = torch.stack(imgs).to(device)
    labels = torch.tensor(labs, dtype=torch.long).to(device)
    idx = torch.randperm(N)
    n_train = int(split[0] * N)
    n_val   = int(split[1] * N)
    return {
        "E": E, "labels": labels,
        "train_dataset": NodeEmbeddingDataset(E, labels, idx[:n_train]),
        "val_dataset":   NodeEmbeddingDataset(E, labels, idx[n_train:n_train + n_val]),
        "test_dataset":  NodeEmbeddingDataset(E, labels, idx[n_train + n_val:]),
        "meta": {"name": "mnist", "n_nodes": N, "n_classes": 10, "feat_dim": 784},
    }


def make_loaders(
    data: dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Return dict of {train, val, test} DataLoaders."""
    return {
        split: DataLoader(
            data[f"{split}_dataset"],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
        for split in ("train", "val", "test")
    }
