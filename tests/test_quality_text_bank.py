"""Five-word quality bank checks; no dataset or model downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from models.two_stage_models import (
    CrossModalIQAFullModel, CrossModalIQAJointPretrain,
    QualityTextPrototypeBank, load_pretrained_backbone,
)
from train_cm_pretrain import _compute_quality_text_loss, _extract_representation_state


CENTERS = [0.375, 0.54166667, 0.66666669, 0.79166666, 0.95833334]


class TinyFrozenText(nn.Module):
    output_dim = 6

    def encode_base_text(self, phrases):
        assert len(phrases) == 5
        assert "terrible" in phrases[0] and "excellent" in phrases[-1]
        return torch.eye(6)[:5]


class TinyDownstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.quality_adapter = nn.Linear(4, 4)
        self.use_quality_adapter = True
        self.quality_text_bank = QualityTextPrototypeBank(TinyFrozenText(), 4, CENTERS)
        self.use_quality_text_bank = True


class TinyBackbone(nn.Module):
    fused_dim = 8
    local_dim = 4
    text_dim = 6

    def __init__(self):
        super().__init__()
        self.text_encoder = TinyFrozenText()
        self.global_encoder = nn.Linear(3, 8)
        self.local_encoder = nn.Conv2d(3, 4, 1)
        self.fusion = nn.Identity()

    def encode_image(self, image, local_crops=None):
        z = self.global_encoder(image.mean(dim=(2, 3)))
        local = self.local_encoder(torch.nn.functional.adaptive_avg_pool2d(image, (2, 2)))
        return {"z": z, "f_loc": local}

    def encode_text(self, prompts, device):
        return torch.ones(len(prompts), 6, device=device)


class QualityTextBankTests(unittest.TestCase):
    def test_fixed_text_prototypes_and_image_gradients(self):
        bank = QualityTextPrototypeBank(TinyFrozenText(), 4, CENTERS)
        image = torch.randn(3, 4, requires_grad=True)
        out = bank(image)
        self.assertEqual(tuple(out["quality_text_logits"].shape), (3, 5))
        self.assertEqual(tuple(out["quality_text_score"].shape), (3,))
        self.assertTrue(bank.text_features.requires_grad)
        self.assertEqual(tuple(bank.text_features.shape), (5, 6))
        self.assertTrue(((out["quality_text_score"] >= CENTERS[0]) &
                         (out["quality_text_score"] <= CENTERS[-1])).all().item())
        target = torch.tensor([0.37, 0.66, 0.96])
        losses = _compute_quality_text_loss(out, target, torch.ones(3), bank.centers)
        self.assertTrue(torch.isfinite(losses["quality_text_ce"]).item())
        losses["quality_text_ce"].backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertGreater(bank.image_to_text.weight.grad.abs().sum().item(), 0)

    def test_soft_targets_are_ordinal_and_weights_are_respected(self):
        centers = torch.tensor(CENTERS)
        target = torch.tensor([CENTERS[0], CENTERS[-1]])
        good = {"quality_text_logits": torch.tensor([[8., 0., 0., 0., 0.],
                                                       [0., 0., 0., 0., 8.]]),
                "quality_text_score": target}
        bad = {"quality_text_logits": good["quality_text_logits"].flip(1),
               "quality_text_score": target.flip(0)}
        self.assertLess(
            _compute_quality_text_loss(good, target, torch.ones(2), centers)["quality_text_ce"],
            _compute_quality_text_loss(bad, target, torch.ones(2), centers)["quality_text_ce"],
        )
        masked = _compute_quality_text_loss(good, target, torch.tensor([1., 0.]), centers)
        first = _compute_quality_text_loss(
            {"quality_text_logits": good["quality_text_logits"][:1],
             "quality_text_score": target[:1]}, target[:1], torch.ones(1), centers
        )
        torch.testing.assert_close(masked["quality_text_ce"], first["quality_text_ce"])
        self.assertEqual(_compute_quality_text_loss({}, target, torch.ones(2), None)["quality_text_ce"].item(), 0.0)

    def test_downstream_blend_changes_only_quality_path(self):
        def config(blend):
            return {
                "model": {
                    "heads": {"mlp_hidden_dim": 8, "dropout": 0.0},
                    "quality_branch_use_text_gate": False,
                    "quality_adapter": {
                        "enabled": True, "output_dim": 8, "hidden_dim": 16,
                        "dropout": 0.0, "last_bn": False,
                    },
                    "quality_text_bank": {
                        "enabled": blend > 0.0, "centers": CENTERS,
                        "temperature": 0.2, "score_blend": blend,
                    },
                }
            }

        with patch("models.two_stage_models.ImageTextBackbone", side_effect=lambda cfg: TinyBackbone()):
            torch.manual_seed(15)
            base = CrossModalIQAFullModel(config(0.0)).eval()
            torch.manual_seed(15)
            blended = CrossModalIQAFullModel(config(0.1)).eval()
        image = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            base_out = base.forward_single(image, prompts=["cat", "dog"])
            blend_out = blended.forward_single(image, prompts=["cat", "dog"])
        torch.testing.assert_close(base_out["s_align"], blend_out["s_align"])
        torch.testing.assert_close(base_out["s_quality"], blend_out["s_quality_visual"])
        torch.testing.assert_close(base_out["s_quality_visual"], blend_out["s_quality_visual"])
        torch.testing.assert_close(
            blend_out["s_quality"],
            0.9 * blend_out["s_quality_visual"] + 0.1 * blend_out["quality_text_score"],
        )

    def test_three_view_text_score_depends_on_clean_view_only(self):
        cfg = {
            "model": {
                "heads": {"mlp_hidden_dim": 8, "dropout": 0.0},
                "quality_branch_use_text_gate": False,
                "quality_adapter": {"enabled": False},
                "quality_text_bank": {
                    "enabled": True, "centers": CENTERS,
                    "temperature": 0.2, "score_blend": 0.0,
                },
                "simsiam": {
                    "proj_dim": 8, "proj_hidden_dim": 16,
                    "pred_hidden_dim": 8, "separate_quality_projection": True,
                },
            }
        }
        with patch("models.two_stage_models.ImageTextBackbone", side_effect=lambda cfg: TinyBackbone()):
            model = CrossModalIQAJointPretrain(cfg).eval()
        clean = torch.randn(2, 3, 16, 16)
        prompts = ["cat", "dog"]
        answers = ["cat", "dog"]
        with torch.no_grad():
            a = model.forward_views(
                torch.randn_like(clean), torch.randn_like(clean),
                prompts, answers, image_clean=clean,
            )
            b = model.forward_views(
                torch.randn_like(clean), torch.randn_like(clean),
                prompts, answers, image_clean=clean,
            )
        torch.testing.assert_close(a["quality_text_score"], b["quality_text_score"])
        torch.testing.assert_close(a["quality_text_logits"], b["quality_text_logits"])
        torch.testing.assert_close(
            a["quality_text_score"],
            model.iqa.quality_text_bank(a["quality_clean_z"])["quality_text_score"],
        )

    def test_slim_checkpoint_keeps_bank_and_loader_requires_it(self):
        model = TinyDownstream()
        state = {
            "iqa.quality_text_bank.text_features": torch.ones(5, 6),
            "iqa.quality_text_bank.centers": torch.tensor(CENTERS),
            "iqa.quality_text_bank.image_to_text.weight": torch.full((6, 4), 0.25),
            "quality_image_projector.weight": torch.eye(4),
            "quality_image_predictor.weight": torch.ones(4, 4),
        }
        compact = _extract_representation_state(state)
        self.assertIn("iqa.quality_text_bank.image_to_text.weight", compact)
        self.assertNotIn("quality_image_predictor.weight", compact)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrain.pth"
            torch.save({"model": compact}, checkpoint)
            load_pretrained_backbone(model, str(checkpoint), load_scope="representation")
            torch.testing.assert_close(model.quality_text_bank.image_to_text.weight,
                                       torch.full((6, 4), 0.25))
            torch.save({"model": {"quality_image_projector.weight": torch.eye(4)}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Quality text bank is enabled"):
                load_pretrained_backbone(TinyDownstream(), str(checkpoint), load_scope="representation")


if __name__ == "__main__":
    unittest.main()
