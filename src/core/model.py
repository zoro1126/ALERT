"""
model.py
========
BiGRU + Scaled Dot-Product Attention model for driver fatigue classification.

Architecture
------------
Input : (batch, seq_len, n_features)  — default (B, 60, 8)
         ↓
BiGRU layer 1  : hidden=64  → output (B, 60, 128)   [64 × 2 directions]
         ↓
Dropout (p=0.3)
         ↓
BiGRU layer 2  : hidden=32  → output (B, 60, 64)    [32 × 2 directions]
         ↓
Attention      : learns a scalar weight per timestep → context (B, 64)
         ↓
Dense + ReLU   : 64 → 64
         ↓
Dropout (p=0.2)
         ↓
Dense (logits) : 64 → n_classes (3)

Output : raw logits (B, 3) — use CrossEntropyLoss during training,
         softmax at inference time.

Design notes
------------
- Bidirectional GRU captures both forward (eyes closing) and backward
  (recovery after microsleep) temporal context.
- Stacked two-layer BiGRU follows Liu et al. (2022) and the systematic
  review (Sensors 2023) showing 2-layer BiRNN >> 1-layer for PERCLOS tasks.
- Additive attention (Bahdanau-style but simplified to a learned weight
  vector) lets the model up-weight the most fatigue-relevant timesteps
  (e.g., the frames inside a microsleep episode) rather than using only
  the final hidden state.
- All parameters: ~130 K — well under the <100 MB on-disk target.

References
----------
- Liu et al. (2022): CNN-LSTM hybrid; we adapt their temporal stage as
  a standalone BiGRU since our input is already a hand-crafted feature vec.
- Bahdanau et al. (2015): attention mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Learned scalar attention over a sequence of GRU hidden states.

    Given input  H : (B, T, hidden_dim)
    Computes     a = softmax(H @ w)          — (B, T) attention weights
    Returns      c = sum_t(a_t * H_t)        — (B, hidden_dim) context vector

    This is a simplified Bahdanau-style attention without a query vector —
    equivalent to learning "which timesteps matter most" globally.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Single linear layer (no bias) projects each hidden state to a score
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : (B, T, hidden_dim)

        Returns
        -------
        context : (B, hidden_dim)
        weights : (B, T)  — useful for visualisation / interpretability
        """
        # score each timestep: (B, T, 1) → (B, T)
        scores = self.score(H).squeeze(-1)
        weights = F.softmax(scores, dim=1)          # (B, T)
        # weighted sum: (B, T, 1) * (B, T, hidden) → (B, hidden)
        context = (weights.unsqueeze(-1) * H).sum(dim=1)
        return context, weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FatigueClassifier(nn.Module):
    """
    Bidirectional GRU + Attention fatigue classifier.

    Parameters
    ----------
    n_features : int   — input feature dimension (default 8)
    hidden1    : int   — BiGRU layer 1 hidden size per direction (default 64)
    hidden2    : int   — BiGRU layer 2 hidden size per direction (default 32)
    fc_size    : int   — fully-connected hidden size (default 64)
    n_classes  : int   — output classes (default 3)
    dropout_rnn: float — dropout between BiGRU layers (default 0.3)
    dropout_fc : float — dropout before logit layer (default 0.2)

    Input  : (batch, seq_len, n_features)
    Output : (batch, n_classes)  — raw logits
    """

    def __init__(
        self,
        n_features : int   = Config.N_FEATURES,
        hidden1    : int   = Config.HIDDEN_SIZE,
        hidden2    : int   = Config.HIDDEN_SIZE_2,
        fc_size    : int   = Config.FC_SIZE,
        n_classes  : int   = Config.N_CLASSES,
        dropout_rnn: float = Config.DROPOUT_RNN,
        dropout_fc : float = Config.DROPOUT_FC,
    ) -> None:
        super().__init__()

        # Layer 1 BiGRU: n_features → hidden1 × 2
        self.bigru1 = nn.GRU(
            input_size=n_features,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop_rnn = nn.Dropout(p=dropout_rnn)

        # Layer 2 BiGRU: hidden1×2 → hidden2 × 2
        self.bigru2 = nn.GRU(
            input_size=hidden1 * 2,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Temporal attention over layer-2 output sequence
        self.attention = TemporalAttention(hidden_dim=hidden2 * 2)

        # Classifier head
        self.fc1      = nn.Linear(hidden2 * 2, fc_size)
        self.drop_fc  = nn.Dropout(p=dropout_fc)
        self.fc_out   = nn.Linear(fc_size, n_classes)

        # Store config for convenience
        self.n_classes  = n_classes
        self.n_features = n_features

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Orthogonal init for GRU weights; Xavier for linear layers."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "fc" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, n_features)

        Returns
        -------
        logits  : (B, n_classes)
        attn_w  : (B, T)  — attention weights (detach before storing)
        """
        # BiGRU 1
        h1, _ = self.bigru1(x)           # (B, T, hidden1*2)
        h1 = self.drop_rnn(h1)

        # BiGRU 2
        h2, _ = self.bigru2(h1)          # (B, T, hidden2*2)

        # Attention → context vector
        ctx, attn_w = self.attention(h2)  # (B, hidden2*2), (B, T)

        # Classifier head
        out = F.relu(self.fc1(ctx))       # (B, fc_size)
        out = self.drop_fc(out)
        logits = self.fc_out(out)         # (B, n_classes)

        return logits, attn_w

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns softmax probabilities (B, n_classes)."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
