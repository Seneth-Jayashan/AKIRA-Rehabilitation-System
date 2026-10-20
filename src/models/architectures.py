"""Unified implementations of the 4 non-negotiable architectures: TCN, 1D CNN, LSTM, and GRU.

All four models accept input tensors of shape (batch_size, window_len, in_channels)
and support:
  - task_mode='classification': outputs logits of shape (batch_size, num_classes)
  - task_mode='regression': outputs continuous angles of shape (batch_size, window_len, num_reg_targets)
  - task_mode='multitask': outputs dict {'logits': (B, C), 'angles': (B, W, D)}
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """Dilated residual 1D convolutional block for Temporal Convolutional Network (TCN)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(F.gelu(self.bn2(self.conv2(out))))
        return F.gelu(out + res)


class TaskOutputHead(nn.Module):
    """Shared task head supporting classification, sequence regression, or masked multi-task output."""

    def __init__(
        self,
        feat_dim: int,
        task_mode: str = "classification",
        num_classes: int = 10,
        num_reg_targets: int = 2,
    ) -> None:
        super().__init__()
        if task_mode not in ("classification", "regression", "multitask"):
            raise ValueError(f"Unsupported task_mode: {task_mode!r}")
        self.task_mode = task_mode

        if task_mode in ("classification", "multitask"):
            self.cls_head = nn.Linear(feat_dim * 2, num_classes)
        if task_mode in ("regression", "multitask"):
            self.reg_head = nn.Linear(feat_dim, num_reg_targets)

    def forward(self, seq_features: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Project sequence features of shape (B, W, D) to task outputs."""
        if self.task_mode == "classification":
            avg_pool = seq_features.mean(dim=1)
            max_pool = seq_features.amax(dim=1)
            return self.cls_head(torch.cat([avg_pool, max_pool], dim=-1))

        if self.task_mode == "regression":
            return self.reg_head(seq_features)

        avg_pool = seq_features.mean(dim=1)
        max_pool = seq_features.amax(dim=1)
        return {
            "logits": self.cls_head(torch.cat([avg_pool, max_pool], dim=-1)),
            "angles": self.reg_head(seq_features),
        }


class TCNModel(nn.Module):
    """Architecture 1: Temporal Convolutional Network (TCN)."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: Sequence[int] = (56, 56, 56, 56),
        kernel_size: int = 5,
        dropout: float = 0.2,
        task_mode: str = "classification",
        num_classes: int = 10,
        num_reg_targets: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        ch_prev = in_channels
        for i, ch_out in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_prev, ch_out, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            ch_prev = ch_out
        self.network = nn.Sequential(*layers)
        self.head = TaskOutputHead(ch_prev, task_mode=task_mode, num_classes=num_classes, num_reg_targets=num_reg_targets)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, W, C) -> transpose to (B, C, W) for Conv1d
        feat = self.network(x.transpose(1, 2)).transpose(1, 2)
        return self.head(feat)


class CNN1DModel(nn.Module):
    """Architecture 2: One-Dimensional Convolutional Neural Network (1D CNN)."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: Sequence[int] = (64, 64, 96, 96),
        kernel_sizes: Sequence[int] = (7, 5, 5, 3),
        dropout: float = 0.2,
        task_mode: str = "classification",
        num_classes: int = 10,
        num_reg_targets: int = 2,
    ) -> None:
        super().__init__()
        blocks = []
        ch_prev = in_channels
        for ch_out, k in zip(hidden_channels, kernel_sizes):
            pad = k // 2
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(ch_prev, ch_out, kernel_size=k, padding=pad, dilation=1),
                    nn.BatchNorm1d(ch_out),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            ch_prev = ch_out
        self.network = nn.Sequential(*blocks)
        self.head = TaskOutputHead(ch_prev, task_mode=task_mode, num_classes=num_classes, num_reg_targets=num_reg_targets)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        feat = self.network(x.transpose(1, 2)).transpose(1, 2)
        return self.head(feat)


class LSTMModel(nn.Module):
    """Architecture 3: Long Short-Term Memory (LSTM) Network."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_size: int = 80,
        num_layers: int = 2,
        dropout: float = 0.2,
        task_mode: str = "classification",
        num_classes: int = 10,
        num_reg_targets: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = TaskOutputHead(hidden_size, task_mode=task_mode, num_classes=num_classes, num_reg_targets=num_reg_targets)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        h = F.gelu(self.input_proj(x))
        out, _ = self.rnn(h)
        out = self.norm(out)
        return self.head(out)


class GRUModel(nn.Module):
    """Architecture 4: Gated Recurrent Unit (GRU) Network."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_size: int = 92,
        num_layers: int = 2,
        dropout: float = 0.2,
        task_mode: str = "classification",
        num_classes: int = 10,
        num_reg_targets: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = TaskOutputHead(hidden_size, task_mode=task_mode, num_classes=num_classes, num_reg_targets=num_reg_targets)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        h = F.gelu(self.input_proj(x))
        out, _ = self.rnn(h)
        out = self.norm(out)
        return self.head(out)


def build_four_models(
    in_channels: int = 12,
    task_mode: str = "classification",
    num_classes: int = 10,
    num_reg_targets: int = 2,
) -> Dict[str, nn.Module]:
    """Instantiate the exact four benchmark architectures with matched ~90k-110k parameter budgets."""
    return {
        "TCN": TCNModel(
            in_channels=in_channels,
            task_mode=task_mode,
            num_classes=num_classes,
            num_reg_targets=num_reg_targets,
        ),
        "1D_CNN": CNN1DModel(
            in_channels=in_channels,
            task_mode=task_mode,
            num_classes=num_classes,
            num_reg_targets=num_reg_targets,
        ),
        "LSTM": LSTMModel(
            in_channels=in_channels,
            task_mode=task_mode,
            num_classes=num_classes,
            num_reg_targets=num_reg_targets,
        ),
        "GRU": GRUModel(
            in_channels=in_channels,
            task_mode=task_mode,
            num_classes=num_classes,
            num_reg_targets=num_reg_targets,
        ),
    }


def compute_masked_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    y_cls: torch.Tensor,
    mask_cls: torch.Tensor,
    y_angles: torch.Tensor,
    mask_angles: torch.Tensor,
    lambda_angles: float = 1.0,
) -> torch.Tensor:
    """Compute masked multi-task loss where unavailable targets have mask == 0."""
    loss = torch.tensor(0.0, device=outputs["logits"].device)
    if mask_cls.any():
        cls_losses = F.cross_entropy(outputs["logits"], y_cls, reduction="none")
        loss = loss + (cls_losses * mask_cls.float()).sum() / mask_cls.float().sum().clamp_min(1.0)
    if mask_angles.any():
        reg_losses = F.smooth_l1_loss(outputs["angles"], y_angles, reduction="none").mean(dim=(1, 2))
        loss = loss + lambda_angles * (reg_losses * mask_angles.float()).sum() / mask_angles.float().sum().clamp_min(1.0)
    return loss
