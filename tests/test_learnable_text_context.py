"""Learnable CLIP text-prefix and representation-transfer checks (no downloads)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from models.two_stage_models import FrozenCLIPTextEncoder, load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import build_optimizer


class TinyTokenizer:
    def __call__(self, texts, context_length=None):
        length = context_length or 8
        result = torch.zeros(len(texts), length, dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [1] + [2 + (ord(char) % 5) for char in text] + [9]
            ids = ids[:length]
            ids[-1] = 9
            result[row, :len(ids)] = torch.tensor(ids)
        return result


class TinyTransformer(nn.Module):
    def get_cast_dtype(self):
        return torch.float32

    def forward(self, x, attn_mask=None):
        return x.cumsum(dim=1)


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_length = 8
        self.token_embedding = nn.Embedding(16, 4)
        self.positional_embedding = nn.Parameter(torch.randn(8, 4) * 0.01)
        self.transformer = TinyTransformer()
        self.attn_mask = None
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(4))
        self.text_pool_type = "argmax"
        self.visual = nn.Linear(4, 4)

    def encode_text(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = x[torch.arange(x.size(0)), tokens.argmax(dim=-1)]
        return x @ self.text_projection


class TinyDownstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.text_encoder = nn.Module()
        self.backbone.text_encoder.context_vectors = nn.Parameter(torch.zeros(4, 4))
        self.text_to_fused = nn.Linear(4, 4)
        self.use_quality_adapter = False


class LearnableTextContextTests(unittest.TestCase):
    def make_encoder(self, count, isolate_rng=False):
        toy = TinyCLIP()
        with (
            patch("open_clip.create_model_and_transforms", return_value=(toy, None, None)),
            patch("open_clip.get_tokenizer", return_value=TinyTokenizer()),
        ):
            return FrozenCLIPTextEncoder(
                pretrained=None, learnable_context_tokens=count,
                isolate_context_init_rng=isolate_rng,
            )

    def test_disabled_context_preserves_frozen_path(self):
        encoder = self.make_encoder(0)
        result = encoder(["cat", "dog"], torch.device("cpu"))
        self.assertEqual(result.shape, (2, 4))
        self.assertFalse(result.requires_grad)
        self.assertIsNone(encoder.context_vectors)
        self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))

    def test_isolated_prefix_preserves_later_rng_draws(self):
        torch.manual_seed(431)
        self.make_encoder(0)
        expected = torch.rand(5)

        torch.manual_seed(431)
        isolated = self.make_encoder(4, isolate_rng=True)
        actual = torch.rand(5)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(isolated.context_vectors.shape), (4, 4))

        torch.manual_seed(431)
        self.make_encoder(4, isolate_rng=False)
        legacy = torch.rand(5)
        self.assertFalse(torch.equal(legacy, expected))

    def test_context_receives_gradient_including_auxiliary_text_path(self):
        encoder = self.make_encoder(4)
        result = encoder(["cat", "dog"], torch.device("cpu"))
        auxiliary = encoder.encode_cached(["cat"], torch.device("cpu"))
        (result[:, 0].sum() + auxiliary[:, 1].sum()).backward()
        self.assertEqual(result.shape, (2, 4))
        self.assertIsNotNone(encoder.context_vectors.grad)
        self.assertGreater(encoder.context_vectors.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in encoder.model.parameters()))
        self.assertEqual(len(encoder._feature_cache), 0)

    def test_context_uses_dedicated_low_lr_downstream(self):
        model = TinyDownstream()
        optimizer = build_optimizer(
            {"train": {"lr": 1e-4, "weight_decay": 0.0,
                       "layerwise_lr": {"enabled": True}}}, model
        )
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertAlmostEqual(groups["text_context"]["lr"], 1e-5)
        self.assertEqual(
            [id(param) for param in groups["text_context"]["params"]],
            [id(model.backbone.text_encoder.context_vectors)],
        )

    def test_compact_pretrain_checkpoint_retains_context(self):
        state = {
            "iqa.backbone.text_encoder.context_vectors": torch.ones(4, 4),
            "iqa.text_to_fused.weight": torch.ones(4, 4),
            "text_projector.weight": torch.ones(4, 4),
        }
        compact = _extract_representation_state(state)
        self.assertIn("iqa.backbone.text_encoder.context_vectors", compact)
        self.assertIn("iqa.text_to_fused.weight", compact)
        self.assertNotIn("text_projector.weight", compact)

    def test_representation_loader_restores_context_and_rejects_mismatch(self):
        model = TinyDownstream()
        correct = torch.randn(4, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrain.pth"
            torch.save({"model": {
                "iqa.backbone.text_encoder.context_vectors": correct,
                "iqa.text_to_fused.weight": torch.ones(4, 4),
            }}, checkpoint)
            load_pretrained_backbone(
                model, str(checkpoint), load_scope="representation"
            )
            self.assertTrue(torch.equal(model.backbone.text_encoder.context_vectors, correct))
            torch.save({"model": {
                "iqa.backbone.text_encoder.context_vectors": torch.ones(3, 4),
            }}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "compatible context_vectors"):
                load_pretrained_backbone(
                    TinyDownstream(), str(checkpoint), load_scope="representation"
                )


if __name__ == "__main__":
    unittest.main()
