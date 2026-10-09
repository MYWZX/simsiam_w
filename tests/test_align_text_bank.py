"""Five-word image-Prompt matching bank checks; no dataset or downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from models.two_stage_models import (
    CrossModalIQAFullModel, CrossModalIQAJointPretrain, load_pretrained_backbone,
)
from train_cm_pretrain import _compute_align_text_loss, _extract_representation_state


CENTERS = [0.0, 0.25, 0.5, 0.75, 1.0]


class TinyFrozenText(nn.Module):
    output_dim = 6

    def encode_base_text(self, phrases):
        assert len(phrases) == 5
        assert "image and its prompt" in phrases[0]
        assert "terrible" in phrases[0] and "excellent" in phrases[-1]
        return torch.eye(6)[:5]


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
        out = torch.zeros(len(prompts), 6, device=device)
        for i, prompt in enumerate(prompts):
            out[i, 0 if "cat" in prompt else 1] = 1.0
        return out


def config(blend=0.0, bank=True):
    return {
        "model": {
            "heads": {"mlp_hidden_dim": 8, "dropout": 0.0},
            "quality_branch_use_text_gate": False,
            "quality_adapter": {"enabled": False},
            "align_text_bank": {
                "enabled": bank, "centers": CENTERS,
                "temperature": 0.2, "score_blend": blend,
            },
            "simsiam": {
                "proj_dim": 8, "proj_hidden_dim": 16,
                "pred_hidden_dim": 8, "separate_quality_projection": True,
            },
        },
    }


class AlignTextBankTests(unittest.TestCase):
    def test_blend_changes_only_align_output_and_uses_real_prompt(self):
        with patch("models.two_stage_models.ImageTextBackbone", side_effect=lambda cfg: TinyBackbone()):
            torch.manual_seed(21)
            base = CrossModalIQAFullModel(config(bank=False)).eval()
            torch.manual_seed(21)
            bank = CrossModalIQAFullModel(config(blend=0.1)).eval()
        image = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            original = base.forward_single(image, prompts=["cat eats"])
            cat = bank.forward_single(image, prompts=["cat eats"])
            dog = bank.forward_single(image, prompts=["dog eats"])
            changed_image = bank.forward_single(image + 1.0, prompts=["cat eats"])
        torch.testing.assert_close(original["s_quality"], cat["s_quality"])
        torch.testing.assert_close(original["s_align"], cat["s_align_base"])
        torch.testing.assert_close(
            cat["s_align"], 0.9 * cat["s_align_base"] + 0.1 * cat["align_text_score"]
        )
        self.assertGreater(
            (cat["align_text_logits"] - dog["align_text_logits"]).abs().max().item(), 1e-7
        )
        self.assertGreater(
            (cat["align_text_logits"] - changed_image["align_text_logits"]).abs().max().item(), 1e-7
        )

    def test_three_view_bank_uses_clean_view_only(self):
        with patch("models.two_stage_models.ImageTextBackbone", side_effect=lambda cfg: TinyBackbone()):
            model = CrossModalIQAJointPretrain(config()).eval()
        clean = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            a = model.forward_views(
                torch.randn_like(clean), torch.randn_like(clean),
                ["cat eats", "dog eats"], ["cat", "dog"], image_clean=clean,
            )
            b = model.forward_views(
                torch.randn_like(clean), torch.randn_like(clean),
                ["cat eats", "dog eats"], ["cat", "dog"], image_clean=clean,
            )
        torch.testing.assert_close(a["align_text_score"], b["align_text_score"])
        torch.testing.assert_close(a["align_text_logits"], b["align_text_logits"])

    def test_low_align_labels_are_not_downweighted(self):
        centers = torch.tensor(CENTERS)
        target = torch.tensor([CENTERS[0], CENTERS[-1]])
        good = {
            "align_text_logits": torch.tensor([
                [8., 0., 0., 0., 0.], [0., 0., 0., 0., 8.],
            ]),
            "align_text_score": target,
        }
        wrong = {
            "align_text_logits": good["align_text_logits"].flip(1),
            "align_text_score": target.flip(0),
        }
        self.assertLess(
            _compute_align_text_loss(good, target, centers)["align_text_ce"],
            _compute_align_text_loss(wrong, target, centers)["align_text_ce"],
        )
        low_wrong = {
            "align_text_logits": torch.tensor([[0., 0., 0., 0., 8.]]),
            "align_text_score": target[-1:],
        }
        low_loss = _compute_align_text_loss(low_wrong, target[:1], centers)["align_text_ce"]
        high_wrong = _compute_align_text_loss(
            {"align_text_logits": torch.tensor([[8., 0., 0., 0., 0.]]),
             "align_text_score": target[:1]},
            target[-1:], centers,
        )["align_text_ce"]
        torch.testing.assert_close(low_loss, high_wrong)

    def test_slim_checkpoint_keeps_bank_and_loader_requires_it(self):
        with patch("models.two_stage_models.ImageTextBackbone", side_effect=lambda cfg: TinyBackbone()):
            model = CrossModalIQAFullModel(config()).eval()
        state = {
            "iqa.align_text_bank.text_features": torch.eye(6)[:5],
            "iqa.align_text_bank.centers": torch.tensor(CENTERS),
            "iqa.align_text_bank.image_to_text.weight": torch.full((6, 24), 0.25),
        }
        compact = _extract_representation_state(state)
        self.assertIn("iqa.align_text_bank.image_to_text.weight", compact)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrain.pth"
            torch.save({"model": compact}, checkpoint)
            load_pretrained_backbone(model, str(checkpoint), load_scope="representation")
            torch.testing.assert_close(
                model.align_text_bank.image_to_text.weight, torch.full((6, 24), 0.25)
            )
            torch.save({"model": {}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Align text bank is enabled"):
                load_pretrained_backbone(model, str(checkpoint), load_scope="representation")


if __name__ == "__main__":
    unittest.main()
