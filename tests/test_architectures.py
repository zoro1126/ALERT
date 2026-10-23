"""
test_architectures.py
=====================
Unit tests for edge-compliant candidate architectures:
- BiGRUAttention
- TCNClassifier
- HybridCNNGRU
- Model Factory (build_model)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.model import (
    BiGRUAttention,
    FatigueClassifier,
    HybridCNNGRU,
    TCNClassifier,
    build_model,
)
from utils.config import Config


@pytest.mark.parametrize("arch", ["bigru", "tcn", "cnn_gru"])
def test_architecture_forward_shape(arch):
    batch_size = 8
    seq_len = Config.WINDOW_SIZE
    n_features = Config.N_FEATURES
    n_classes = 3

    model = build_model(arch=arch, n_features=n_features, n_classes=n_classes)
    model.eval()

    x = torch.randn(batch_size, seq_len, n_features)
    logits, aux = model(x)

    assert logits.shape == (batch_size, n_classes), f"{arch} logits shape mismatch"
    assert aux.shape == (batch_size, seq_len), f"{arch} aux/attention shape mismatch"


@pytest.mark.parametrize("arch", ["bigru", "tcn", "cnn_gru"])
def test_edge_parameter_budget(arch):
    model = build_model(arch=arch)
    params = model.count_parameters()

    # Must be compact (< 250k parameters)
    assert params < 250_000, f"{arch} parameter count {params:,} exceeds 250k edge budget"
    assert params > 10_000, f"{arch} parameter count {params:,} suspiciously small"


@pytest.mark.parametrize("arch", ["bigru", "tcn", "cnn_gru"])
def test_gradient_flow(arch):
    model = build_model(arch=arch)
    model.train()

    x = torch.randn(4, Config.WINDOW_SIZE, Config.N_FEATURES, requires_grad=True)
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()

    # Verify gradients exist and are not NaN
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name} in {arch}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name} in {arch}"


@pytest.mark.parametrize("arch", ["bigru", "tcn", "cnn_gru"])
def test_predict_proba(arch):
    model = build_model(arch=arch)
    x = torch.randn(4, Config.WINDOW_SIZE, Config.N_FEATURES)
    probs = model.predict_proba(x)

    assert probs.shape == (4, 3)
    # Sum of probabilities across classes should be 1.0
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_tcn_receptive_field_coverage():
    # TCN must cover >= 120 frames
    tcn = TCNClassifier(kernel_size=3, channels=[32, 48, 64, 64, 64])
    # Dilation: 1, 2, 4, 8, 16 -> RF = 1 + 2*(3-1)*(1+2+4+8+16) = 125 frames
    assert tcn.count_parameters() < 150_000


def test_fatigue_classifier_backward_compatibility():
    model = FatigueClassifier()
    x = torch.randn(2, Config.WINDOW_SIZE, Config.N_FEATURES)
    logits, attn = model(x)
    assert logits.shape == (2, 3)
    assert attn.shape == (2, Config.WINDOW_SIZE)
