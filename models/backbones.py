# vit+resnet、本地/全局视觉编码器
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _require_timm():
    try:
        import timm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "timm is required for ImageNet/Hugging Face pretrained backbones. "
            "Please run `pip install timm huggingface_hub safetensors`."
        ) from e
    return timm


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + identity)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleViTEncoder(nn.Module):
    """Fallback lightweight ViT, used only when you explicitly choose simple_vit."""
    def __init__(self, output_dim: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, embed_dim=output_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.norm = nn.LayerNorm(output_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        b = tokens.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        feat = self.encoder(tokens)
        feat = self.norm(feat[:, 0])
        return feat


class GlobalVisualEncoder(nn.Module):
    """
    Recommended:
    - encoder_type='vit_base_patch16_224.augreg2_in21k_ft_in1k'
    - pretrained=True
    These weights are fetched by timm, which typically resolves them from Hugging Face Hub.
    """
    def __init__(
        self,
        encoder_type: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained: bool = True,
        output_dim: int = 768,
        img_size: int = 224,
        open_clip_model_name: str = "ViT-B-16",
        open_clip_pretrained: str = "openai",
        source_normalization: str = "imagenet",
        normalize_output: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.source_normalization = str(source_normalization).strip().lower()
        self.normalize_output = bool(normalize_output)
        self._uses_open_clip = encoder_type == "open_clip"

        if encoder_type == "simple_vit":
            self.backbone = SimpleViTEncoder(output_dim=output_dim, img_size=img_size)
            self.feat_dim = output_dim
        elif encoder_type == "open_clip":
            try:
                import open_clip
            except ImportError as e:
                raise ImportError("open_clip_torch is not installed. Run `pip install open_clip_torch`.") from e
            model, _, _ = open_clip.create_model_and_transforms(
                open_clip_model_name,
                pretrained=open_clip_pretrained if pretrained else None,
            )
            self.backbone = model.visual
            self.feat_dim = getattr(model.visual, "output_dim", 512)

            # Project datasets emit ImageNet-normalized tensors because the
            # same input is also consumed by the local ResNet. Convert only the
            # global branch back to [0, 1], then apply this OpenCLIP model's
            # expected normalization. Merely changing the YAML would otherwise
            # feed the pretrained visual tower the wrong input distribution.
            preprocess_cfg = open_clip.get_model_preprocess_cfg(model)
            clip_mean = preprocess_cfg.get(
                "mean", (0.48145466, 0.4578275, 0.40821073)
            )
            clip_std = preprocess_cfg.get(
                "std", (0.26862954, 0.26130258, 0.27577711)
            )
            self.register_buffer(
                "source_mean",
                torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "source_std",
                torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "model_mean",
                torch.tensor(clip_mean, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "model_std",
                torch.tensor(clip_std, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            timm = _require_timm()
            self.backbone = timm.create_model(
                encoder_type,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                img_size=img_size,
            )
            self.feat_dim = getattr(self.backbone, "num_features", output_dim)

        self.output_dim = output_dim
        self.proj = nn.Linear(self.feat_dim, output_dim) if self.feat_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._uses_open_clip:
            if self.source_normalization == "imagenet":
                source_mean = self.source_mean.to(dtype=x.dtype)
                source_std = self.source_std.to(dtype=x.dtype)
                x = x * source_std + source_mean
            elif self.source_normalization not in {"none", "zero_one", "0_1"}:
                raise ValueError(
                    "OpenCLIP global encoder source_normalization must be "
                    f"'imagenet' or 'none', got {self.source_normalization!r}."
                )
            model_mean = self.model_mean.to(dtype=x.dtype)
            model_std = self.model_std.to(dtype=x.dtype)
            x = (x.clamp(0.0, 1.0) - model_mean) / model_std
        feat = self.backbone(x)
        feat = self.proj(feat)
        if self.normalize_output:
            feat = torch.nn.functional.normalize(feat.float(), dim=-1).to(feat.dtype)
        return feat


class ResNet50LocalEncoder(nn.Module):
    """
    Recommended:
    - model_name='resnet50.a1_in1k'
    - pretrained=True
    It produces a spatial feature map and uses timm pretrained ImageNet weights.
    """
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 256,
        model_name: str = "resnet50.a1_in1k",
        feature_stage: str = "layer4",
    ):
        super().__init__()
        if feature_stage not in {"layer3", "layer4", "layer3_4"}:
            raise ValueError("local_encoder.feature_stage must be layer3, layer4 or layer3_4.")
        self.feature_stage = feature_stage
        timm = _require_timm()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=((3, 4) if feature_stage == "layer3_4" else ({"layer3": 3, "layer4": 4}[feature_stage],)),
        )
        feat_info = self.backbone.feature_info
        if feature_stage == "layer3_4":
            channels = feat_info.channels()
            self.proj3 = nn.Conv2d(channels[0], output_dim, kernel_size=1)
            self.proj4 = nn.Conv2d(channels[1], output_dim, kernel_size=1)
            self.stage_mix_logit = nn.Parameter(torch.zeros(()))
        else:
            # Preserve existing single-stage state_dict names and initialization.
            self.proj = nn.Conv2d(feat_info.channels()[-1], output_dim, kernel_size=1)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.feature_stage == "layer3_4":
            p3 = self.proj3(feats[0])
            p4 = self.proj4(feats[1])
            p4 = torch.nn.functional.interpolate(
                p4, size=p3.shape[-2:], mode="bilinear", align_corners=False,
            )
            weight3 = torch.sigmoid(self.stage_mix_logit)
            return weight3 * p3 + (1.0 - weight3) * p4
        return self.proj(feats[-1])


def maybe_freeze(module: nn.Module, freeze: bool) -> None:
    if freeze:
        for p in module.parameters():
            p.requires_grad = False
