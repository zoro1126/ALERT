"""
test_model.py
=============
Unit tests for FatigueClassifier and TemporalAttention.

Run:  pytest tests/test_model.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.model import FatigueClassifier, TemporalAttention


# ---------------------------------------------------------------------------
# TemporalAttention
# ---------------------------------------------------------------------------

class TestTemporalAttention:
    def test_output_shapes(self):
        attn = TemporalAttention(hidden_dim=64)
        H = torch.randn(4, 60, 64)
        context, weights = attn(H)
        assert context.shape == (4, 64)
        assert weights.shape == (4, 60)

    def test_weights_sum_to_one(self):
        attn = TemporalAttention(hidden_dim=64)
        H = torch.randn(2, 30, 64)
        _, weights = attn(H)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_single_timestep(self):
        """Edge case: T=1 — attention weight must be exactly 1.0."""
        attn = TemporalAttention(hidden_dim=32)
        H = torch.randn(1, 1, 32)
        _, weights = attn(H)
        assert weights.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# FatigueClassifier
# ---------------------------------------------------------------------------

class TestFatigueClassifier:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return FatigueClassifier(
            n_features=8, hidden1=64, hidden2=32,
            fc_size=64, n_classes=3,
        )

    def test_forward_output_shapes(self, model):
        x = torch.randn(4, 60, 8)
        logits, attn = model(x)
        assert logits.shape == (4, 3)
        assert attn.shape   == (4, 60)

    def test_attention_weights_sum_to_one(self, model):
        x = torch.randn(4, 60, 8)
        _, attn = model(x)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_predict_proba_sums_to_one(self, model):
        x = torch.randn(3, 60, 8)
        probs = model.predict_proba(x)
        assert probs.shape == (3, 3)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_predict_proba_no_grad(self, model):
        """predict_proba should not track gradients."""
        x = torch.randn(2, 60, 8)
        probs = model.predict_proba(x)
        assert not probs.requires_grad

    def test_parameter_count_reasonable(self, model):
        n = model.count_parameters()
        # Expect 50K–200K for this architecture
        assert 50_000 < n < 200_000, f"Unexpected param count: {n}"

    def test_single_sample_batch(self, model):
        """Batch size of 1 must not crash (common real-time case)."""
        x = torch.randn(1, 60, 8)
        logits, attn = model(x)
        assert logits.shape == (1, 3)
        assert attn.shape   == (1, 60)

    def test_different_seq_lengths(self, model):
        """Model should handle seq_len != 60 (though training uses 60)."""
        x = torch.randn(2, 30, 8)
        logits, attn = model(x)
        assert logits.shape == (2, 3)
        assert attn.shape   == (2, 30)

    def test_training_vs_eval_dropout(self, model):
        """
        With dropout active, two forward passes in train mode should differ.
        In eval mode they should be identical.
        """
        x = torch.randn(4, 60, 8)

        model.train()
        out1, _ = model(x)
        out2, _ = model(x)
        # Should differ due to dropout (with very high probability)
        assert not torch.allclose(out1, out2)

        model.eval()
        with torch.no_grad():
            out3, _ = model(x)
            out4, _ = model(x)
        assert torch.allclose(out3, out4)

    def test_gradients_flow(self, model):
        """Loss backward should populate all parameter gradients."""
        model.train()
        x = torch.randn(4, 60, 8)
        y = torch.randint(0, 3, (4,))
        logits, _ = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"

    def test_save_load_roundtrip(self, model, tmp_path):
        """State dict save → load → identical outputs."""
        model.eval()
        x = torch.randn(2, 60, 8)
        with torch.no_grad():
            out_before, _ = model(x)

        save_path = tmp_path / "test_model.pt"
        torch.save({"model_state": model.state_dict()}, save_path)

        model2 = FatigueClassifier(n_features=8, hidden1=64, hidden2=32,
                                   fc_size=64, n_classes=3)
        ckpt = torch.load(save_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        with torch.no_grad():
            out_after, _ = model2(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)
