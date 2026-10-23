"""
losses.py
=========
Production-grade loss functions for imbalanced driver fatigue classification.

Key Objectives
--------------
1. Focal Loss (Lin et al., 2017):
   Addresses extreme class imbalance and hard boundary ambiguity by down-weighting
   easy examples (e.g. clearly alert or obvious microsleep) and focusing gradients
   on hard, subtle fatigue states (Class 1: Low Vigilant).

   FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

2. Label Smoothing (Szegedy et al., 2016):
   Prevents overconfidence on transition frames where fatigue states naturally overlap.

3. Class-Weighted Cross Entropy:
   Penalizes false negatives on Class 1 (Low Vigilant) and Class 2 (Drowsy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional class weighting and label smoothing.

    Parameters
    ----------
    alpha : torch.Tensor | Sequence[float] | None
        Per-class weights of shape (C,). If None, all classes have equal weight 1.0.
    gamma : float
        Focusing parameter (default 2.0). Higher gamma reduces the loss for easy
        examples, shifting gradient attention to ambiguous/hard samples.
    label_smoothing : float
        Label smoothing factor between 0.0 and 1.0 (default 0.05).
    reduction : str
        'mean', 'sum', or 'none' (default 'mean').
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) unnormalized raw predictions.
        targets : (B,) integer ground truth class indices.

        Returns
        -------
        loss : scalar tensor if reduction in ('mean', 'sum'), else (B,).
        """
        num_classes = logits.size(-1)

        # Log-softmax for numerical stability
        log_p = F.log_softmax(logits, dim=-1)           # (B, C)
        p = log_p.exp()                                 # (B, C)

        # Extract probabilities for the true classes
        # p_t: probability of true class
        p_t = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B,)

        # Compute focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - p_t, self.gamma)                  # (B,)

        # Cross entropy component with optional label smoothing
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )                                                                # (B,)

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class OrdinalFocalLoss(nn.Module):
    """
    Ordinal Distance-Penalized Focal Loss (master_plan.md Section 3.2).

    L = L_Focal + lambda_ord * L_Ordinal
    where L_Ordinal = sum_{c=0}^{C-1} p_c * (c - y_true)^2
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.08,
        lambda_ord: float = 0.35,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        self.lambda_ord = lambda_ord
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(logits, targets)  # (B,)

        probs = F.softmax(logits, dim=-1)  # (B, C)
        num_classes = logits.size(-1)

        classes = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)  # (C,)
        diff = classes.unsqueeze(0) - targets.unsqueeze(-1).to(logits.dtype)            # (B, C)
        ord_penalty = (probs * torch.pow(diff, 2)).sum(dim=-1)                         # (B,)

        total_loss = focal + self.lambda_ord * ord_penalty

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        elif self.reduction == "none":
            return total_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


def get_criterion(
    loss_name: str = "ordinal",
    class_weights: torch.Tensor | list[float] | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.08,
    lambda_ord: float = 0.35,
) -> nn.Module:
    """
    Factory helper to instantiate the chosen loss criterion.
    """
    loss_name = loss_name.lower()
    if loss_name in ("ordinal", "ordinal_focal"):
        return OrdinalFocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
            lambda_ord=lambda_ord,
        )
    elif loss_name == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    elif loss_name in ("ce", "cross_entropy"):
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss type '{loss_name}'. Use 'ordinal', 'focal', or 'ce'.")
