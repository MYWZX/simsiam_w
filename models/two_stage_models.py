# 预训练模型,当前核心模型文件，包含预训练模型和主 IQA 模型
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import GlobalVisualEncoder, ResNet50LocalEncoder, maybe_freeze
from .fusion import SemanticStructureFusion, AlignOnlyVMambaFusion


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, last_bn: bool = False):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        ]
        if last_bn:
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualAdapter(nn.Module):
    """Small residual MLP adapter used to specialize features for one branch."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        learnable_context_tokens: int = 0,
        isolate_context_init_rng: bool = False,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("open_clip_torch is required: pip install open_clip_torch") from e

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        with torch.no_grad():
            tok = self.tokenizer(["test"])
            feat = self.model.encode_text(tok)
        self.output_dim = int(feat.shape[-1])
        self.num_context_tokens = int(learnable_context_tokens)
        if not 0 <= self.num_context_tokens <= 16:
            raise ValueError("learnable_context_tokens must be in [0, 16].")
        if self.num_context_tokens:
            # Only prefix embeddings are learned; the entire CLIP model stays
            # frozen. Leave space for SOT and EOT even for long user prompts.
            context_length = int(self.model.context_length)
            if self.num_context_tokens > context_length - 2:
                raise ValueError("Learnable context leaves no room for SOT and EOT.")
            width = int(self.model.token_embedding.embedding_dim)
            if isolate_context_init_rng:
                # Do not shift the initialization of later heads in matched ablations.
                with torch.random.fork_rng(devices=[]):
                    initial_context = torch.empty(self.num_context_tokens, width).normal_(std=0.02)
            else:
                # Preserve the initialization of existing context-token runs.
                initial_context = torch.empty(self.num_context_tokens, width).normal_(std=0.02)
            self.context_vectors = nn.Parameter(initial_context)
        else:
            self.register_parameter("context_vectors", None)
        # The global visual encoder is created separately. The frozen text path
        # never calls OpenCLIP's visual tower, so discard it to reduce memory and
        # checkpoint size while preserving encode_text/tokenizer behavior.
        if hasattr(self.model, "visual"):
            self.model.visual = nn.Identity()
        self._feature_cache: Dict[str, torch.Tensor] = {}

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        self.model.eval()
        strings = [str(t) for t in texts]
        if not self.num_context_tokens:
            with torch.no_grad():
                tokens = self.tokenizer(strings).to(device)
                feat = self.model.encode_text(tokens)
            return F.normalize(feat.float(), dim=-1)

        from open_clip.transformer import text_global_pool

        short_tokens = self.tokenizer(
            strings,
            context_length=int(self.model.context_length) - self.num_context_tokens,
        ).to(device)
        # Placeholder IDs preserve the shifted EOT position for CLIP pooling.
        placeholders = torch.zeros(
            short_tokens.size(0), self.num_context_tokens,
            dtype=short_tokens.dtype, device=device,
        )
        token_ids = torch.cat(
            [short_tokens[:, :1], placeholders, short_tokens[:, 1:]], dim=1
        )
        cast_dtype = self.model.transformer.get_cast_dtype()
        fixed_embeddings = self.model.token_embedding(short_tokens).to(cast_dtype)
        context = self.context_vectors.to(cast_dtype).unsqueeze(0).expand(
            short_tokens.size(0), -1, -1
        )
        x = torch.cat(
            [fixed_embeddings[:, :1], context, fixed_embeddings[:, 1:]], dim=1
        )
        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)
        feat = text_global_pool(
            x, token_ids, self.model.text_pool_type,
            eos_token_id=getattr(self.model, "text_eos_id", None),
        )
        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                feat = self.model.text_projection(feat)
            else:
                feat = feat @ self.model.text_projection
        return F.normalize(feat.float(), dim=-1)

    def encode_with_context(self, texts: Sequence[str], context_vectors: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
        """Encode one independently learnable context per text prompt.

        ``context_vectors`` has shape ``[batch, N, width]``.  The frozen CLIP
        text transformer still runs, so gradients reach only these inserted
        context tokens and never the CLIP parameters.
        """
        from open_clip.transformer import text_global_pool

        self.model.eval()
        if context_vectors.ndim != 3 or context_vectors.size(0) != len(texts):
            raise ValueError("context_vectors must have shape [len(texts), N, width].")
        n_tokens = int(context_vectors.size(1))
        if n_tokens < 1 or n_tokens > int(self.model.context_length) - 2:
            raise ValueError("Invalid number of class-specific context tokens.")
        short_tokens = self.tokenizer(
            [str(t) for t in texts],
            context_length=int(self.model.context_length) - n_tokens,
        ).to(device)
        placeholders = torch.zeros(
            short_tokens.size(0), n_tokens, dtype=short_tokens.dtype, device=device
        )
        token_ids = torch.cat([short_tokens[:, :1], placeholders, short_tokens[:, 1:]], dim=1)
        cast_dtype = self.model.transformer.get_cast_dtype()
        fixed_embeddings = self.model.token_embedding(short_tokens).to(cast_dtype)
        x = torch.cat([
            fixed_embeddings[:, :1],
            context_vectors.to(device=device, dtype=cast_dtype),
            fixed_embeddings[:, 1:],
        ], dim=1)
        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)
        feat = text_global_pool(
            x, token_ids, self.model.text_pool_type,
            eos_token_id=getattr(self.model, "text_eos_id", None),
        )
        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                feat = self.model.text_projection(feat)
            else:
                feat = feat @ self.model.text_projection
        return F.normalize(feat.float(), dim=-1)

    def encode_token_features(
        self, texts: Sequence[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen CLIP token features and token ids for cross-attention.

        The returned features are before CLIP's pooled text projection and have
        shape ``[B, context_length, text_width]``. Padding ids are returned so
        downstream attention can ignore them. Learnable prefix contexts remain
        differentiable when enabled; CLIP parameters stay frozen.
        """
        self.model.eval()
        strings = [str(t) for t in texts]
        if self.num_context_tokens:
            short_tokens = self.tokenizer(
                strings,
                context_length=int(self.model.context_length) - self.num_context_tokens,
            ).to(device)
            placeholders = torch.zeros(
                short_tokens.size(0), self.num_context_tokens,
                dtype=short_tokens.dtype, device=device,
            )
            token_ids = torch.cat(
                [short_tokens[:, :1], placeholders, short_tokens[:, 1:]], dim=1
            )
            cast_dtype = self.model.transformer.get_cast_dtype()
            fixed_embeddings = self.model.token_embedding(short_tokens).to(cast_dtype)
            context = self.context_vectors.to(cast_dtype).unsqueeze(0).expand(
                short_tokens.size(0), -1, -1
            )
            x = torch.cat(
                [fixed_embeddings[:, :1], context, fixed_embeddings[:, 1:]], dim=1
            )
        else:
            token_ids = self.tokenizer(strings).to(device)
            cast_dtype = self.model.transformer.get_cast_dtype()
            x = self.model.token_embedding(token_ids).to(cast_dtype)
        x = x + self.model.positional_embedding.to(x.dtype)
        with torch.set_grad_enabled(bool(self.num_context_tokens)):
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            x = self.model.ln_final(x)
        return x.float(), token_ids

    def encode_cached(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        """Encode short frozen-CLIP texts once and cache their CPU features."""
        if self.num_context_tokens:
            # Cached detached features would silently stop prompt learning.
            return self.forward(texts, device=device)
        keys = [str(text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self._feature_cache))
        if missing:
            features = self.forward(missing, device=device).cpu()
            for key, feature in zip(missing, features):
                self._feature_cache[key] = feature
        return torch.stack([self._feature_cache[key] for key in keys], dim=0).to(
            device=device, non_blocking=True
        )

    @torch.no_grad()
    def encode_base_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode fixed quality descriptors without learned Prompt/Answer context."""
        self.model.eval()
        device = next(self.model.parameters()).device
        tokens = self.tokenizer([str(value) for value in texts]).to(device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)


class QualityTextPrototypeBank(nn.Module):
    """Five independent trainable text-token classifiers.

    Each level is initialized from its complete CLIP phrase, then its own
    text-space prototype is optimized independently.  The CLIP text encoder
    remains frozen; only these five classifier tokens and ``image_to_text``
    are trainable.
    """

    WORDS = ("terrible", "poor", "fair", "good", "excellent")

    def __init__(self, text_encoder: FrozenCLIPTextEncoder, image_dim: int,
                 centers: Sequence[float], temperature: float = 0.2,
                 phrases: Optional[Sequence[str]] = None,
                 class_context_tokens: int = 0):
        super().__init__()
        values = [float(value) for value in centers]
        if len(values) != len(self.WORDS) or not all(0.0 <= value <= 1.0 for value in values) or not all(
            left < right for left, right in zip(values, values[1:])
        ):
            raise ValueError("quality_text_bank.centers must contain five increasing values in [0, 1].")
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("quality_text_bank.temperature must be positive.")
        if phrases is None:
            phrases = [f"the quality of the AI-generated image is {word}" for word in self.WORDS]
        if len(phrases) != len(self.WORDS):
            raise ValueError("Five-level text bank requires exactly five phrases.")
        object.__setattr__(self, "_phrases", tuple(str(p) for p in phrases))
        self.class_context_tokens = int(class_context_tokens)
        if self.class_context_tokens:
            width = int(text_encoder.model.token_embedding.embedding_dim)
            shared = getattr(text_encoder, "context_vectors", None)
            if shared is not None and shared.size(0) == self.class_context_tokens:
                initial = shared.detach().unsqueeze(0).expand(len(self.WORDS), -1, -1).clone()
                initial += 0.001 * torch.randn_like(initial)
            else:
                initial = torch.empty(len(self.WORDS), self.class_context_tokens, width)
                initial.normal_(std=0.02)
            self.class_context = nn.Parameter(initial)
            self.register_buffer("initial_class_context", initial.detach().clone())
            with torch.no_grad():
                initial_features = text_encoder.encode_with_context(
                    phrases, self.class_context.detach(), device=next(text_encoder.model.parameters()).device
                )
        else:
            self.register_parameter("class_context", None)
            self.register_buffer("initial_class_context", torch.empty(0))
            initial_features = text_encoder.encode_base_text(phrases)
        # Kept as an initialization/reference buffer for compatibility and
        # diagnostics. In class-context mode forward recomputes text features
        # through the frozen CLIP transformer so class_context receives grads.
        self.register_buffer("text_features", initial_features)
        self.register_buffer("centers", torch.tensor(values, dtype=torch.float32))
        self.image_to_text = nn.Linear(int(image_dim), text_encoder.output_dim, bias=False)
        object.__setattr__(self, "_text_encoder_ref", text_encoder)

    def context_anchor_loss(self) -> torch.Tensor:
        """Keep learned class prompts close to their CLIP initialization."""
        if self.class_context is None or self.initial_class_context.numel() == 0:
            return self.centers.new_zeros(())
        return (self.class_context - self.initial_class_context).pow(2).mean()

    def forward(self, image_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_text = F.normalize(self.image_to_text(image_feature).float(), dim=-1)
        if self.class_context is not None:
            references = self._text_encoder_ref.encode_with_context(
                self._phrases, self.class_context, device=image_feature.device
            )
        else:
            references = self.text_features
        references = F.normalize(references.float(), dim=-1)
        logits = image_text @ references.T / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "quality_text_logits": logits,
            "quality_text_score": probabilities @ self.centers,
        }


class ImageTextBackbone(nn.Module):
    """Shared encoders used by both stage-1 pretraining and stage-2 IQA fine-tuning."""

    def __init__(self, cfg: Dict):
        super().__init__()
        mcfg = cfg["model"]
        gcfg = mcfg["global_encoder"]
        lcfg = mcfg["local_encoder"]
        fcfg = mcfg["fusion"]
        tcfg = mcfg.get("text_encoder", {})

        self.global_encoder = GlobalVisualEncoder(
            encoder_type=gcfg.get("type", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
            pretrained=gcfg.get("pretrained", True),
            output_dim=gcfg.get("output_dim", 768),
            img_size=cfg["dataset"].get("input_size", 224),
            open_clip_model_name=gcfg.get(
                "model_name", tcfg.get("clip_model_name", "ViT-B-16")
            ),
            open_clip_pretrained=gcfg.get(
                "open_clip_pretrained", tcfg.get("clip_pretrained", "openai")
            ),
            source_normalization=gcfg.get("source_normalization", "imagenet"),
            normalize_output=gcfg.get("normalize_output", False),
        )
        self.local_encoder = ResNet50LocalEncoder(
            pretrained=lcfg.get("pretrained", True),
            output_dim=lcfg.get("output_dim", 256),
            model_name=lcfg.get("type", "resnet50.a1_in1k"),
        )
        maybe_freeze(self.global_encoder, gcfg.get("freeze", False))
        maybe_freeze(self.local_encoder, lcfg.get("freeze", False))

        self.fusion = SemanticStructureFusion(
            semantic_dim=gcfg.get("output_dim", 768),
            local_dim=lcfg.get("output_dim", 256),
            embed_dim=fcfg.get("embed_dim", 256),
            num_heads=fcfg.get("num_heads", 4),
            dropout=fcfg.get("dropout", 0.1),
        )
        self.fused_dim = int(fcfg.get("embed_dim", 256))
        self.local_dim = int(lcfg.get("output_dim", 256))
        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=tcfg.get("clip_model_name", "ViT-B-16"),
            pretrained=tcfg.get("clip_pretrained", "openai"),
            learnable_context_tokens=tcfg.get("learnable_context_tokens", 0),
            isolate_context_init_rng=tcfg.get("isolate_context_init_rng", False),
        )
        self.text_dim = self.text_encoder.output_dim

    def encode_image(
        self,
        x: torch.Tensor,
        local_crops: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        f_sem = self.global_encoder(x)    # [B, D_sem]
        if local_crops is None:
            f_loc = self.local_encoder(x)  # [B, C, H, W]
        else:
            if local_crops.ndim != 5:
                raise ValueError(
                    "local_crops must have shape [B,K,3,H,W], "
                    f"got {tuple(local_crops.shape)}"
                )
            batch_size, num_crops, channels, crop_h, crop_w = local_crops.shape
            if batch_size != x.shape[0] or channels != x.shape[1]:
                raise ValueError(
                    "local_crops batch/channel dimensions must match image: "
                    f"image={tuple(x.shape)}, local_crops={tuple(local_crops.shape)}"
                )
            flat_crops = local_crops.reshape(
                batch_size * num_crops, channels, crop_h, crop_w
            )
            crop_maps = self.local_encoder(flat_crops)
            _, local_channels, local_h, local_w = crop_maps.shape
            # Concatenate crop feature maps along the spatial-height axis. The
            # fusion and quality heads then attend over every crop token while
            # retaining each crop's within-patch spatial layout.
            f_loc = (
                crop_maps.reshape(
                    batch_size, num_crops, local_channels, local_h, local_w
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(batch_size, local_channels, num_crops * local_h, local_w)
            )
        z = self.fusion(f_sem, f_loc)     # [B, E]
        return {
            "f_sem": f_sem,
            "f_loc": f_loc,
            "z": z,
            "num_local_crops": 1 if local_crops is None else int(local_crops.shape[1]),
        }

    def encode_text(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        return self.text_encoder(texts, device=device)


class CrossModalSimSiamPretrain(nn.Module):
    """Stage 1: pure cross-modal SimSiam pretraining.

    The model only outputs representations; MOS-aware ranking is applied in train_cm_pretrain.py.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.backbone = ImageTextBackbone(cfg)
        mcfg = cfg["model"]
        pcfg = mcfg.get("simsiam", {})
        hcfg = mcfg.get("heads", {})
        dropout = float(hcfg.get("dropout", 0.1))
        proj_dim = int(pcfg.get("proj_dim", 256))
        proj_hidden = int(pcfg.get("proj_hidden_dim", 512))
        pred_hidden = int(pcfg.get("pred_hidden_dim", 128))
        quality_probe_hidden = int(pcfg.get("quality_probe_hidden_dim", hcfg.get("mlp_hidden_dim", 256)))
        self.use_quality_probe = bool(cfg.get("train", {}).get("use_dual_proxy_supervision", False))

        self.image_projector = MLP(self.backbone.fused_dim, proj_hidden, proj_dim, dropout=dropout, last_bn=True)
        self.text_projector = MLP(self.backbone.text_dim, proj_hidden, proj_dim, dropout=dropout, last_bn=True)
        self.image_predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )
        self.text_predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )
        # Pretraining-only probe. It is intentionally outside `backbone.*`, so
        # downstream checkpoint loading discards the probe while retaining the
        # quality-sensitive gradients learned by the shared visual backbone.
        self.quality_probe = nn.Sequential(
            nn.Linear(self.backbone.fused_dim, quality_probe_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(quality_probe_hidden, 1),
        )

    def _project_image(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_proj = F.normalize(self.image_projector(z), dim=-1)
        p = F.normalize(self.image_predictor(z_proj), dim=-1)
        return {"z_proj": z_proj, "p": p}

    def _project_text(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_proj = F.normalize(self.text_projector(feat), dim=-1)
        p = F.normalize(self.text_predictor(z_proj), dim=-1)
        return {"z_proj": z_proj, "p": p}

    def forward_views(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        prompts: Sequence[str],
        answers: Sequence[str],
        image_clean: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        img1 = self.backbone.encode_image(image1)
        img2 = self.backbone.encode_image(image2)
        img1_p = self._project_image(img1["z"])
        img2_p = self._project_image(img2["z"])
        prompt_feat = self.backbone.encode_text(prompts, image1.device)
        answer_feat = self.backbone.encode_text(answers, image1.device)
        prompt_p = self._project_text(prompt_feat)
        answer_p = self._project_text(answer_feat)
        out = {
            "img1_z": img1_p["z_proj"], "img1_p": img1_p["p"],
            "img2_z": img2_p["z_proj"], "img2_p": img2_p["p"],
            "prompt_z": prompt_p["z_proj"], "prompt_p": prompt_p["p"],
            "answer_z": answer_p["z_proj"], "answer_p": answer_p["p"],
        }
        clean = None
        if image_clean is not None:
            clean = self.backbone.encode_image(image_clean)
            clean_p = self._project_image(clean["z"])
            out.update({"clean_z": clean_p["z_proj"], "clean_p": clean_p["p"]})
        if self.use_quality_probe:
            quality_pred1 = torch.sigmoid(self.quality_probe(img1["z"])).squeeze(-1)
            quality_pred2 = torch.sigmoid(self.quality_probe(img2["z"])).squeeze(-1)
            quality_pred = 0.5 * (quality_pred1 + quality_pred2)
            if clean is not None:
                quality_pred = torch.sigmoid(self.quality_probe(clean["z"])).squeeze(-1)
            out.update({
                "quality_pred1": quality_pred1,
                "quality_pred2": quality_pred2,
                "quality_pred": quality_pred,
            })
        return out


class CrossModalIQAFullModel(nn.Module):
    """Stage 2: main IQA model.

    Includes:
      1. Quality branch: ResNet local patches + NSS + global prompt text gate.
      2. Alignment branch: global align + prompt-guided patch evidence + gate fusion.
      3. Final score fusion: S_final = w_img*S_quality + (1-w_img)*S_align.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.backbone = ImageTextBackbone(cfg)
        mcfg = cfg["model"]
        hcfg = mcfg.get("heads", {})
        ngcfg = mcfg.get("nss_text_gate", {})
        agcfg = mcfg.get("align_gate", {})
        fcfg = mcfg.get("fusion", {})
        fused_dim = self.backbone.fused_dim
        local_dim = self.backbone.local_dim
        text_dim = self.backbone.text_dim
        hidden_dim = int(hcfg.get("mlp_hidden_dim", 256))
        dropout = float(hcfg.get("dropout", 0.1))
        gate_hidden = int(ngcfg.get("gate_hidden_dim", hidden_dim))
        nss_dim = int(ngcfg.get("nss_dim", 6))

        # Shared text projection to fused dimension.
        self.text_to_fused = nn.Linear(text_dim, fused_dim)

        # Quality branch: local texture map -> patch tokens.
        self.local_patch_proj = nn.Conv2d(local_dim, fused_dim, kernel_size=1)
        self.nss_proj = nn.Sequential(nn.Linear(nss_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, fused_dim))
        self.nss_gate = nn.Sequential(nn.Linear(fused_dim + nss_dim, gate_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_hidden, 1))
        self.text_gate = nn.Sequential(nn.Linear(fused_dim * 2, gate_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_hidden, 1))
        # Quality is a visual-perceptual task. In the decoupled pretraining
        # variant it must not receive a text-conditioned gate. The default
        # remains True for backward compatibility with existing checkpoints.
        self.quality_branch_use_text_gate = bool(
            mcfg.get("quality_branch_use_text_gate", True)
        )
        # Ablation removes only the explicit local-score contribution. The
        # fused global representation still contains ResNet/Fusion information.
        self.quality_global_only = bool(mcfg.get("quality_global_only", False))
        self.patch_quality_head = nn.Sequential(nn.Linear(fused_dim, gate_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_hidden, 1))
        qlcfg = mcfg.get("quality_local_attention", {})
        self.use_quality_local_attention = bool(qlcfg.get("enabled", False))
        if self.use_quality_local_attention:
            q_heads = int(qlcfg.get("num_heads", fcfg.get("num_heads", 4)))
            if fused_dim % q_heads != 0:
                raise ValueError("quality_local_attention.num_heads must divide fusion.embed_dim")
            self.quality_local_attention = nn.MultiheadAttention(
                fused_dim, q_heads, dropout=float(qlcfg.get("dropout", dropout)),
                batch_first=True,
            )
            self.quality_local_attention_norm = nn.LayerNorm(fused_dim)
            self.quality_attention_head = nn.Sequential(
                nn.Linear(fused_dim, gate_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(gate_hidden, 1),
            )
            self.quality_attention_mix_logit = nn.Parameter(
                torch.tensor(float(qlcfg.get("mix_init_logit", -1.1)))
            )
        else:
            self.quality_local_attention = None
            self.quality_local_attention_norm = nn.Identity()
            self.quality_attention_head = None
            self.quality_attention_mix_logit = None
        self.global_quality_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.quality_mix_logit = nn.Parameter(torch.tensor(float(ngcfg.get("quality_mix_init", 0.0))))
        qdcfg = mcfg.get("quality_distribution", {})
        self.use_quality_distribution = bool(qdcfg.get("enabled", False))
        self.quality_distribution_temperature = float(qdcfg.get("temperature", 0.2))
        self.register_buffer(
            "quality_distribution_centers",
            torch.tensor(qdcfg.get("centers", [0.0, 0.25, 0.5, 0.75, 1.0]), dtype=torch.float32),
        )
        if self.use_quality_distribution:
            self.quality_distribution_head = nn.Linear(fused_dim, int(self.quality_distribution_centers.numel()))
        else:
            self.quality_distribution_head = None

        # Optional dataset-specific quality factors. These small heads are
        # supervised only when the current dataset exposes the configured
        # labels. They share the complete persistent quality representation,
        # so their gradients teach the encoder useful quality cues without
        # replacing the dataset-agnostic quality prediction.
        qauxcfg = mcfg.get("quality_auxiliary_heads", {})
        self.use_quality_auxiliary_heads = bool(qauxcfg.get("enabled", False))
        raw_components = qauxcfg.get("components", [])
        if isinstance(raw_components, str):
            raw_components = [raw_components]
        self.quality_auxiliary_components = tuple(
            str(component).strip().lower() for component in raw_components
        )
        if self.use_quality_auxiliary_heads and not self.quality_auxiliary_components:
            raise ValueError(
                "model.quality_auxiliary_heads.enabled=true requires at least one component"
            )
        if len(set(self.quality_auxiliary_components)) != len(self.quality_auxiliary_components):
            raise ValueError("model.quality_auxiliary_heads.components contains duplicates")
        for component in self.quality_auxiliary_components:
            if not component or not component.replace("_", "").isalnum():
                raise ValueError(
                    "Quality auxiliary component names may contain only letters, "
                    f"numbers and underscores, got {component!r}."
                )
        aux_hidden_dim = int(qauxcfg.get("hidden_dim", hidden_dim))
        aux_dropout = float(qauxcfg.get("dropout", dropout))
        self.quality_aux_heads = nn.ModuleDict()
        if self.use_quality_auxiliary_heads:
            quality_context_dim = fused_dim * 3
            self.quality_aux_heads.update({
                component: nn.Sequential(
                    nn.Linear(quality_context_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, max(aux_hidden_dim // 2, 1)),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(max(aux_hidden_dim // 2, 1), 1),
                )
                for component in self.quality_auxiliary_components
            })

        # Optional persistent quality representation. Existing dual-projector
        # checkpoints store the matching module as ``quality_image_projector``;
        # load_pretrained_backbone remaps those weights here. Unlike a SimSiam
        # predictor, this adapter is retained by downstream IQA.
        qacfg = mcfg.get("quality_adapter", {})
        self.use_quality_adapter = bool(qacfg.get("enabled", False))
        self.quality_adapter_mode = str(qacfg.get("mode", "replace")).strip().lower()
        if self.quality_adapter_mode not in {"replace", "parallel_score", "residual_feature"}:
            raise ValueError(
                "model.quality_adapter.mode must be 'replace', 'parallel_score', or 'residual_feature', "
                f"got {self.quality_adapter_mode!r}."
            )
        self.normalize_quality_adapter = bool(qacfg.get("normalize", True))
        self.freeze_quality_adapter = bool(qacfg.get("freeze", False))
        self.detach_quality_adapter_input = bool(qacfg.get("detach_input", False))
        if self.use_quality_adapter:
            quality_adapter_dim = int(qacfg.get("output_dim", fused_dim))
            if quality_adapter_dim != fused_dim:
                raise ValueError(
                    "model.quality_adapter.output_dim must equal fusion.embed_dim "
                    f"({fused_dim}), got {quality_adapter_dim}."
                )
            self.quality_adapter = MLP(
                fused_dim,
                int(qacfg.get("hidden_dim", 512)),
                quality_adapter_dim,
                dropout=float(qacfg.get("dropout", dropout)),
                last_bn=bool(qacfg.get("last_bn", True)),
            )
            if self.quality_adapter_mode == "parallel_score":
                self.quality_adapter_head = nn.Sequential(
                    nn.Linear(quality_adapter_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                gate_init = float(qacfg.get("gate_init", 0.05))
                if not 0.0 < gate_init < 1.0:
                    raise ValueError(
                        "model.quality_adapter.gate_init must be strictly between 0 and 1, "
                        f"got {gate_init}."
                    )
                gate_logit = math.log(gate_init / (1.0 - gate_init))
                self.quality_adapter_gate_logit = nn.Parameter(torch.tensor(gate_logit))
            elif self.quality_adapter_mode == "residual_feature":
                # Reuse the existing optimizer/logging scalar slot, but this
                # weight controls feature addition rather than score blending.
                residual_init = float(qacfg.get("residual_init", 0.1))
                if not math.isfinite(residual_init) or not 0.0 < residual_init < 1.0:
                    raise ValueError("quality_adapter.residual_init must be finite and in (0, 1).")
                self.quality_adapter_head = None
                self.quality_adapter_gate_logit = nn.Parameter(
                    torch.tensor(math.log(residual_init / (1.0 - residual_init)))
                )
            else:
                self.quality_adapter_head = None
                self.quality_adapter_gate_logit = None

            if self.freeze_quality_adapter:
                for parameter in self.quality_adapter.parameters():
                    parameter.requires_grad = False
                # Preserve the projector BatchNorm statistics learned during
                # pretraining instead of updating them with downstream batch 8.
                self.quality_adapter.eval()
        else:
            self.quality_adapter = nn.Identity()
            self.quality_adapter_head = None
            self.quality_adapter_gate_logit = None

        # Quality-only text references are independent of the generation
        # Prompt's four learnable context tokens and of the Align branch.
        qt_cfg = mcfg.get("quality_text_bank", {})
        self.use_quality_text_bank = bool(qt_cfg.get("enabled", False))
        self.quality_text_blend = float(qt_cfg.get("score_blend", 0.0))
        if self.quality_global_only and self.quality_text_blend != 0.0:
            raise ValueError("quality_global_only requires quality_text_bank.score_blend=0.")
        if not 0.0 <= self.quality_text_blend <= 1.0:
            raise ValueError("model.quality_text_bank.score_blend must be in [0, 1].")
        if self.quality_text_blend > 0.0 and not self.use_quality_adapter:
            raise ValueError("Quality text scoring requires model.quality_adapter.enabled=true.")
        if self.use_quality_text_bank:
            # The optional bank must not change initialization of existing
            # Align heads in paired-seed ablations.
            with torch.random.fork_rng(devices=[]):
                self.quality_text_bank = QualityTextPrototypeBank(
                    self.backbone.text_encoder,
                    image_dim=fused_dim,
                    centers=qt_cfg["centers"],
                    temperature=float(qt_cfg.get("temperature", 0.2)),
                    class_context_tokens=int(qt_cfg.get("class_context_tokens", 0)),
                )
        else:
            self.quality_text_bank = None

        # Alignment branch: global + patch align + gate fusion.
        self.align_global_head = nn.Sequential(
            nn.Linear(fused_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.align_patch_img_proj = nn.Linear(fused_dim, fused_dim)
        self.align_patch_txt_proj = nn.Linear(fused_dim, fused_dim)
        atcfg = mcfg.get("align_prompt_cross_attention", {})
        self.use_align_prompt_cross_attention = bool(atcfg.get("enabled", False))
        if self.use_align_prompt_cross_attention:
            a_heads = int(atcfg.get("num_heads", fcfg.get("num_heads", 4)))
            if fused_dim % a_heads != 0:
                raise ValueError("align_prompt_cross_attention.num_heads must divide fusion.embed_dim")
            self.align_text_token_proj = nn.Linear(text_dim, fused_dim)
            self.align_prompt_cross_attention = nn.MultiheadAttention(
                fused_dim, a_heads, dropout=float(atcfg.get("dropout", dropout)),
                batch_first=True,
            )
            self.align_prompt_cross_norm = nn.LayerNorm(fused_dim)
            self.align_prompt_cross_scale = float(atcfg.get("residual_scale", 1.0))
        else:
            self.align_text_token_proj = None
            self.align_prompt_cross_attention = None
            self.align_prompt_cross_norm = nn.Identity()
            self.align_prompt_cross_scale = 0.0
        self.align_patch_head = nn.Sequential(
            nn.Linear(fused_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.align_gate = nn.Sequential(
            nn.Linear(fused_dim * 2 + 2, int(agcfg.get("gate_hidden_dim", hidden_dim))),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(int(agcfg.get("gate_hidden_dim", hidden_dim)), 1),
        )
        # Optional sample-adaptive quality/alignment fusion. The residual gate
        # starts at zero, so enabling it initially reproduces the fixed scalar
        # fusion weight and only learns per-sample corrections.
        fcfg = mcfg.get("fusion_gate", {})
        self.use_fusion_gate = bool(fcfg.get("enabled", False))
        self.fusion_gate_detach_inputs = bool(fcfg.get("detach_inputs", True))
        self.fusion_gate_input_mode = str(
            fcfg.get("input_mode", "features_scores")
        ).lower()
        if self.use_fusion_gate:
            fusion_gate_hidden = int(fcfg.get("hidden_dim", hidden_dim))
            if self.fusion_gate_input_mode in {"scores", "score_only", "score-only"}:
                fusion_gate_input_dim = 3
            else:
                # Backward-compatible input used by the original sample gate.
                fusion_gate_input_dim = fused_dim * 2 + 2
            self.fusion_gate = nn.Sequential(
                nn.Linear(fusion_gate_input_dim, fusion_gate_hidden),
                nn.GELU(),
                nn.Dropout(float(fcfg.get("dropout", dropout))),
                nn.Linear(fusion_gate_hidden, 1),
            )
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)
        else:
            self.fusion_gate = None
        self.fusion_logit = nn.Parameter(torch.tensor(float(mcfg.get("fusion_init_logit", 0.0))))

        # Optional align adapter. It specializes a residual copy of the fused
        # image/text features for S_align without changing the quality branch.
        aadcfg = mcfg.get("align_adapter", {})
        self.use_align_adapter = bool(aadcfg.get("enabled", False))
        self.align_adapter_on_text = bool(aadcfg.get("adapt_text", False))
        if self.use_align_adapter:
            adapter_hidden = int(aadcfg.get("hidden_dim", hidden_dim))
            adapter_dropout = float(aadcfg.get("dropout", dropout))
            adapter_scale = float(aadcfg.get("residual_scale", 0.1))
            self.align_global_adapter = ResidualAdapter(fused_dim, adapter_hidden, adapter_dropout, adapter_scale)
            self.align_patch_adapter = ResidualAdapter(fused_dim, adapter_hidden, adapter_dropout, adapter_scale)
            self.align_text_adapter = (
                ResidualAdapter(fused_dim, adapter_hidden, adapter_dropout, adapter_scale)
                if self.align_adapter_on_text else nn.Identity()
            )
        else:
            self.align_global_adapter = nn.Identity()
            self.align_patch_adapter = nn.Identity()
            self.align_text_adapter = nn.Identity()

        # Optional: alignment-only V-Mamba-style spatial mixer.
        # It is used only in the alignment patch evidence branch. The quality
        # branch still uses the original patch tokens, which prevents the
        # quality degradation observed when replacing the whole fusion module.
        avcfg = mcfg.get("align_vmamba", {})
        self.use_align_vmamba = bool(avcfg.get("enabled", False))
        if self.use_align_vmamba:
            self.align_vmamba = AlignOnlyVMambaFusion(
                dim=fused_dim,
                depth=int(avcfg.get("depth", 1)),
                dropout=float(avcfg.get("dropout", dropout)),
                expansion=int(avcfg.get("expansion", 2)),
                kernel_size=int(avcfg.get("kernel_size", 3)),
                residual_scale=float(avcfg.get("residual_scale", 0.5)),
            )
        else:
            self.align_vmamba = None

        # Optional consistency-level text references. The feature compared
        # with them contains both image evidence and the real Prompt.
        at_cfg = mcfg.get("align_text_bank", {})
        self.use_align_text_bank = bool(at_cfg.get("enabled", False))
        self.align_text_blend = float(at_cfg.get("score_blend", 0.0))
        if not 0.0 <= self.align_text_blend <= 1.0:
            raise ValueError("model.align_text_bank.score_blend must be in [0, 1].")
        if self.use_align_text_bank:
            phrases = [
                f"the match between the image and its prompt is {word}"
                for word in QualityTextPrototypeBank.WORDS
            ]
            with torch.random.fork_rng(devices=[]):
                self.align_text_bank = QualityTextPrototypeBank(
                    self.backbone.text_encoder,
                    image_dim=3 * fused_dim,
                    centers=at_cfg["centers"],
                    temperature=float(at_cfg.get("temperature", 0.2)),
                    phrases=phrases,
                    class_context_tokens=int(at_cfg.get("class_context_tokens", 0)),
                )
        else:
            self.align_text_bank = None

    def train(self, mode: bool = True):
        """Keep a frozen pretrained quality projector in evaluation mode."""
        super().train(mode)
        if self.use_quality_adapter and self.freeze_quality_adapter:
            self.quality_adapter.eval()
        return self

    @staticmethod
    def _normalize_img01(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0.0, 1.0)

    def _compute_nss_features(self, image: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        """Fast differentiable NSS-like patch descriptors.

        Uses MSCN-inspired normalized luminance statistics. It is not a slow exact
        scipy GGD fit; alpha/sigma are lightweight proxies suitable for GPU training.
        Output: [B, H*W, 6]
        """
        x = self._normalize_img01(image)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        mu = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        sigma = torch.sqrt(F.avg_pool2d((gray - mu) ** 2, kernel_size=7, stride=1, padding=3) + 1e-6)
        mscn = (gray - mu) / (sigma + 1e-6)
        pooled_mean = F.adaptive_avg_pool2d(mscn, out_hw)
        pooled_var = F.adaptive_avg_pool2d(mscn ** 2, out_hw)
        pooled_abs = F.adaptive_avg_pool2d(mscn.abs(), out_hw)
        dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        grad = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        pooled_grad = F.adaptive_avg_pool2d(grad, out_hw)
        alpha_proxy = pooled_abs / (torch.sqrt(pooled_var + 1e-6) + 1e-6)
        sigma_proxy = pooled_var
        feat = torch.cat([pooled_mean, pooled_var, pooled_abs, pooled_grad, alpha_proxy, sigma_proxy], dim=1)
        return feat.flatten(2).transpose(1, 2)

    def _make_patch_tokens(self, f_loc: torch.Tensor) -> torch.Tensor:
        p = self.local_patch_proj(f_loc)
        return p.flatten(2).transpose(1, 2)  # [B,N,E]

    def _compute_local_nss_features(
        self,
        image: torch.Tensor,
        f_loc: torch.Tensor,
        local_crops: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Align NSS descriptors with whole-image or concatenated crop tokens."""
        out_h, out_w = f_loc.shape[-2:]
        if local_crops is None:
            return self._compute_nss_features(image, (out_h, out_w))

        batch_size, num_crops, channels, crop_h, crop_w = local_crops.shape
        if out_h % num_crops != 0:
            raise ValueError(
                "Concatenated local feature height must be divisible by the "
                f"number of crops: height={out_h}, crops={num_crops}"
            )
        per_crop_h = out_h // num_crops
        flat_crops = local_crops.reshape(
            batch_size * num_crops, channels, crop_h, crop_w
        )
        crop_nss = self._compute_nss_features(flat_crops, (per_crop_h, out_w))
        return crop_nss.reshape(batch_size, num_crops, -1, 6).reshape(
            batch_size, -1, 6
        )

    def _quality_residual_feature(self, shared: torch.Tensor, adapted: torch.Tensor):
        """Add an adapted direction at a fraction of the shared feature norm.

        Fusion uses LayerNorm (norm roughly sqrt(D)), while the pretrained
        adapter is usually L2 normalized (norm 1). Match these scales so alpha
        is meaningful. Only the norm estimate is detached, not either branch.
        """
        alpha = torch.sigmoid(self.quality_adapter_gate_logit)
        magnitude = shared.float().norm(dim=-1, keepdim=True).detach()
        delta = F.normalize(adapted.float(), dim=-1) * magnitude
        fused = shared.float() + alpha * delta
        return fused.to(shared.dtype), alpha

    def _quality_from_patches(
        self,
        image: torch.Tensor,
        z_global: torch.Tensor,
        f_loc: torch.Tensor,
        text_fused: Optional[torch.Tensor],
        local_crops: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict visual quality from local texture patches without global-local cross-attention.

        This is the rollback version that previously performed better.

        Steps:
          1. ResNet local feature map -> patch tokens.
          2. Compute local NSS descriptors and inject them into patch tokens.
          3. NSS gate + Text gate produce patch weights.
          4. Patch scores are weighted pooled and mixed with global score.

        Note:
          We intentionally do NOT use Global-Local Cross-Attention here, because
          experiments showed that the cross-attention quality branch reduced the
          final/align performance.
        """
        patch_tokens = self._make_patch_tokens(f_loc)
        b, n, e = patch_tokens.shape
        h, w = f_loc.shape[-2:]

        # NSS features use the same whole image or native-resolution crops as
        # the ResNet tokens, so every descriptor stays spatially aligned.
        nss = self._compute_local_nss_features(image, f_loc, local_crops)
        patch_tokens = patch_tokens + self.nss_proj(nss)

        quality_attention_score = None
        quality_attention_weights = None
        if self.quality_local_attention is not None:
            # Use the global image representation as a query over local
            # patches. This learns which high-resolution regions matter for
            # quality instead of averaging all crop tokens equally.
            local_query, quality_attention_weights = self.quality_local_attention(
                z_global.unsqueeze(1), patch_tokens, patch_tokens,
                need_weights=True,
            )
            local_query = self.quality_local_attention_norm(
                z_global.unsqueeze(1) + local_query
            ).squeeze(1)
            quality_attention_score = torch.sigmoid(
                self.quality_attention_head(local_query).squeeze(-1)
            )

        # NSS gate: which patches are quality/statistics-sensitive?
        nss_logits = self.nss_gate(torch.cat([patch_tokens, nss], dim=-1)).squeeze(-1)

        # Text gate: which patches are relevant to the global prompt semantics?
        if text_fused is None:
            text_logits = torch.zeros_like(nss_logits)
        else:
            t = text_fused.unsqueeze(1).expand(-1, n, -1)
            text_logits = self.text_gate(torch.cat([patch_tokens, t], dim=-1)).squeeze(-1)

        weights = torch.softmax(nss_logits + text_logits, dim=1)
        patch_scores = torch.sigmoid(self.patch_quality_head(patch_tokens).squeeze(-1))
        s_patch = torch.sum(weights * patch_scores, dim=1)
        if quality_attention_score is not None:
            attention_mix = torch.sigmoid(self.quality_attention_mix_logit)
            s_patch = (
                (1.0 - attention_mix) * s_patch
                + attention_mix * quality_attention_score
            )
        else:
            attention_mix = z_global.new_tensor(0.0)

        # The optional quality adapter receives the learned SimSiam projector
        # weights. ``replace`` preserves the previous behavior. The safer
        # ``parallel_score`` mode retains the complete shared-feature path and
        # lets a small gate use the frozen quality projector only when useful.
        z_quality_global = z_global
        quality_adapter_gate = z_global.new_tensor(0.0)
        if self.use_quality_adapter:
            adapter_input = z_global.detach() if self.detach_quality_adapter_input else z_global
            z_quality_global = self.quality_adapter(adapter_input)
            if self.normalize_quality_adapter:
                z_quality_global = F.normalize(z_quality_global, dim=-1)

            if self.quality_adapter_mode == "parallel_score":
                s_global_shared = torch.sigmoid(
                    self.global_quality_head(z_global).squeeze(-1)
                )
                s_global_adapter = torch.sigmoid(
                    self.quality_adapter_head(z_quality_global).squeeze(-1)
                )
                quality_adapter_gate = torch.sigmoid(self.quality_adapter_gate_logit)
                s_global = (
                    (1.0 - quality_adapter_gate) * s_global_shared
                    + quality_adapter_gate * s_global_adapter
                )
            elif self.quality_adapter_mode == "residual_feature":
                quality_residual, quality_adapter_gate = self._quality_residual_feature(
                    z_global, z_quality_global
                )
                s_global = torch.sigmoid(self.global_quality_head(quality_residual).squeeze(-1))
                # Diagnostics use the SAME head in eval mode so its dropout
                # does not consume RNG or change the actual training path.
                # The five-level auxiliary bank still receives adapter-only
                # features, keeping its input identical to the baseline.
                head_training = self.global_quality_head.training
                try:
                    self.global_quality_head.eval()
                    s_global_shared = torch.sigmoid(self.global_quality_head(z_global).squeeze(-1))
                    s_global_adapter = torch.sigmoid(self.global_quality_head(z_quality_global).squeeze(-1))
                finally:
                    self.global_quality_head.train(head_training)
            else:
                s_global = torch.sigmoid(
                    self.global_quality_head(z_quality_global).squeeze(-1)
                )
                s_global_shared = s_global
                s_global_adapter = s_global
        else:
            s_global_shared = torch.sigmoid(
                self.global_quality_head(z_global).squeeze(-1)
            )
            s_global_adapter = s_global_shared
            s_global = s_global_shared
        if self.quality_global_only:
            mix = z_global.new_tensor(0.0)
            s_quality = s_global
        else:
            mix = torch.sigmoid(self.quality_mix_logit)
            s_quality = mix * s_patch + (1.0 - mix) * s_global
        s_quality_visual = s_quality
        quality_distribution_out = {}
        if self.quality_distribution_head is not None:
            quality_distribution_logits = self.quality_distribution_head(z_quality_global)
            quality_distribution_prob = torch.softmax(
                quality_distribution_logits / max(self.quality_distribution_temperature, 1e-6), dim=-1
            )
            quality_distribution_out = {
                "quality_distribution_logits": quality_distribution_logits,
                "quality_distribution_score": quality_distribution_prob @ self.quality_distribution_centers,
            }
        quality_text_out = {}
        if self.quality_text_bank is not None:
            quality_text_out = self.quality_text_bank(z_quality_global)
            if self.quality_text_blend > 0.0:
                s_quality = (
                    (1.0 - self.quality_text_blend) * s_quality_visual
                    + self.quality_text_blend * quality_text_out["quality_text_score"]
                )

        result = {
            "s_quality": s_quality,
            "s_quality_visual": s_quality_visual,
            **quality_distribution_out,
            **quality_text_out,
            "s_quality_patch": s_patch,
            "s_quality_global": s_global,
            "s_quality_global_shared": s_global_shared,
            "s_quality_global_adapter": s_global_adapter,
            "quality_adapter_gate": quality_adapter_gate.detach(),
            "quality_mix": mix.detach(),
            "quality_patch_weights": weights,
            "quality_attention_weights": quality_attention_weights,
            "quality_attention_score": quality_attention_score,
            "quality_attention_mix": attention_mix.detach(),
            "patch_tokens": patch_tokens,
            "patch_hw": (h, w),
            "z_quality_global": z_quality_global,
        }
        if self.use_quality_auxiliary_heads:
            patch_evidence = torch.sum(weights.unsqueeze(-1) * patch_tokens, dim=1)
            quality_context = torch.cat(
                [
                    z_quality_global,
                    patch_evidence,
                    torch.abs(z_quality_global - patch_evidence),
                ],
                dim=-1,
            )
            for component, head in self.quality_aux_heads.items():
                result[f"quality_aux_{component}_pred"] = torch.sigmoid(
                    head(quality_context).squeeze(-1)
                )
        return result

    def _align_from_global_and_patch(
        self,
        z_global: torch.Tensor,
        patch_tokens: torch.Tensor,
        text_fused: torch.Tensor,
        patch_hw: Optional[tuple[int, int]] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_align = self.align_global_adapter(z_global)
        text_align = self.align_text_adapter(text_fused)
        patch_align = self.align_patch_adapter(patch_tokens)

        align_cross_attention_weights = None
        if (
            self.align_prompt_cross_attention is not None
            and text_tokens is not None
        ):
            token_features = self.align_text_token_proj(text_tokens)
            key_padding_mask = None
            if text_token_ids is not None:
                key_padding_mask = text_token_ids.eq(0)
            cross_out, align_cross_attention_weights = self.align_prompt_cross_attention(
                patch_align,
                token_features,
                token_features,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            patch_align = self.align_prompt_cross_norm(
                patch_align + self.align_prompt_cross_scale * cross_out
            )

        pair_g = torch.cat([z_align, text_align, torch.abs(z_align - text_align)], dim=-1)
        s_global = torch.sigmoid(self.align_global_head(pair_g).squeeze(-1))

        # Align-only V-Mamba enhancement. Do not feed this back to the quality
        # branch. This preserves quality features while improving local visual
        # evidence search for image-text alignment.
        if self.align_vmamba is not None:
            patch_tokens_for_align = self.align_vmamba(patch_align, hw=patch_hw)
        else:
            patch_tokens_for_align = patch_align

        p_img = F.normalize(self.align_patch_img_proj(patch_tokens_for_align), dim=-1)
        p_txt = F.normalize(self.align_patch_txt_proj(text_align), dim=-1)
        attn = torch.softmax(torch.sum(p_img * p_txt.unsqueeze(1), dim=-1), dim=1)
        evidence = torch.sum(attn.unsqueeze(-1) * patch_tokens_for_align, dim=1)
        pair_p = torch.cat([evidence, text_align, torch.abs(evidence - text_align)], dim=-1)
        s_patch = torch.sigmoid(self.align_patch_head(pair_p).squeeze(-1))

        gate_in = torch.cat([z_align, text_align, s_global.unsqueeze(-1), s_patch.unsqueeze(-1)], dim=-1)
        gate_logit = self.align_gate(gate_in).squeeze(-1)
        g = torch.sigmoid(gate_logit)
        s_align = g * s_global + (1.0 - g) * s_patch
        s_align_base = s_align
        align_text_out = {}
        if self.align_text_bank is not None:
            # Pair features already include the Prompt; use the same learned
            # global/patch gate as the existing alignment prediction.
            pair_fused = g.unsqueeze(-1) * pair_g + (1.0 - g).unsqueeze(-1) * pair_p
            bank_out = self.align_text_bank(pair_fused)
            align_text_out = {
                "align_text_logits": bank_out["quality_text_logits"],
                "align_text_score": bank_out["quality_text_score"],
            }
            if self.align_text_blend > 0.0:
                s_align = (
                    (1.0 - self.align_text_blend) * s_align_base
                    + self.align_text_blend * align_text_out["align_text_score"]
                )
        return {
            "s_align": s_align,
            "s_align_base": s_align_base,
            **align_text_out,
            "s_align_global": s_global,
            "s_align_patch": s_patch,
            "align_gate": g,
            "align_gate_logit": gate_logit,
            "align_gate_input": gate_in,
            "align_patch_weights": attn,
            "align_cross_attention_weights": align_cross_attention_weights,
            # Exposed only for RichHF relational pretraining. They are the
            # same projected features used to form downstream patch evidence.
            "align_patch_features": p_img,
            "align_text_feature": p_txt,
        }

    def _compute_fusion_weight(
        self,
        z_global: torch.Tensor,
        text_fused: torch.Tensor,
        s_quality: torch.Tensor,
        s_align: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return quality weight and per-sample gate residual."""
        if not self.use_fusion_gate:
            return torch.sigmoid(self.fusion_logit), z_global.new_zeros(z_global.size(0))
        if self.fusion_gate_input_mode in {"scores", "score_only", "score-only"}:
            gate_input = torch.stack(
                [s_quality, s_align, torch.abs(s_quality - s_align)], dim=-1
            )
        else:
            gate_input = torch.cat(
                [z_global, text_fused, s_quality.unsqueeze(-1), s_align.unsqueeze(-1)],
                dim=-1,
            )
        if self.fusion_gate_detach_inputs:
            gate_input = gate_input.detach()
        gate_delta = self.fusion_gate(gate_input).squeeze(-1)
        gate_logit = self.fusion_logit + gate_delta
        return torch.sigmoid(gate_logit), gate_delta

    def forward_single(
        self,
        image: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        local_crops: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Keep high-resolution local crops isolated to the Quality branch.
        # Align continues to use the base-resolution shared visual path.
        img = self.backbone.encode_image(image)
        quality_img = (
            self.backbone.encode_image(image, local_crops=local_crops)
            if local_crops is not None else img
        )
        text_feat = None
        text_fused = None
        text_tokens = None
        text_token_ids = None
        if prompts is not None:
            text_feat = self.backbone.encode_text(prompts, image.device)
            text_fused = self.text_to_fused(text_feat)
            if self.align_prompt_cross_attention is not None:
                text_tokens, text_token_ids = self.backbone.text_encoder.encode_token_features(
                    prompts, image.device
                )

        quality_text_fused = (
            text_fused if self.quality_branch_use_text_gate else None
        )
        q_out = self._quality_from_patches(
            image,
            quality_img["z"],
            quality_img["f_loc"],
            quality_text_fused,
            local_crops=local_crops,
        )
        out = {**img, **q_out}

        if prompts is None:
            out["s_align"] = torch.zeros_like(q_out["s_quality"])
            out["s_final"] = q_out["s_quality"]
            out["s_q"] = q_out["s_quality"]
            return out

        align_patch_tokens = self._make_patch_tokens(img["f_loc"])
        a_out = self._align_from_global_and_patch(
            img["z"], align_patch_tokens, text_fused,
            patch_hw=(img["f_loc"].shape[-2], img["f_loc"].shape[-1]),
            text_tokens=text_tokens,
            text_token_ids=text_token_ids,
        )
        w_img, fusion_gate_delta = self._compute_fusion_weight(
            img["z"], text_fused, q_out["s_quality"], a_out["s_align"]
        )
        s_final = w_img * q_out["s_quality"] + (1.0 - w_img) * a_out["s_align"]
        out.update(a_out)
        out.update({
            "text_feat": text_feat,
            "text_fused": text_fused,
            "text_tokens": text_tokens,
            "text_token_ids": text_token_ids,
            "s_final": s_final,
            "s_q": s_final,
            "fusion_w_img": w_img.detach(),
            "fusion_gate_delta": fusion_gate_delta.detach(),
            # Keep the logit for an optional supervised fusion-gate loss.
            "fusion_gate_logit": (
                torch.logit(w_img.clamp(1e-6, 1.0 - 1e-6))
                if self.use_fusion_gate
                else self.fusion_logit.expand_as(w_img)
            ),
        })
        return out

    def predict_all_scores(
        self,
        image: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        local_crops: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_single(image, prompts=prompts, local_crops=local_crops)

    def predict_score(
        self,
        image: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        local_crops: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_single(
            image, prompts=prompts, local_crops=local_crops
        )["s_final"]


class CrossModalIQAJointPretrain(nn.Module):
    """Stage 1 model that keeps the downstream IQA heads in the checkpoint.

    The full IQA model learns RichHF quality/alignment/final supervision while
    disposable SimSiam projectors and predictors provide relational pretraining.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.iqa = CrossModalIQAFullModel(cfg)
        mcfg = cfg["model"]
        pcfg = mcfg.get("simsiam", {})
        hcfg = mcfg.get("heads", {})
        dropout = float(hcfg.get("dropout", 0.1))
        proj_dim = int(pcfg.get("proj_dim", 256))
        proj_hidden = int(pcfg.get("proj_hidden_dim", 512))
        pred_hidden = int(pcfg.get("pred_hidden_dim", 128))
        fused_dim = self.iqa.backbone.fused_dim

        # The original image/text heads form the semantic SimSiam space. An
        # optional second image-only head forms a quality-specific SimSiam
        # space. Both are disposable: only ``iqa`` is loaded downstream.
        self.image_projector = MLP(fused_dim, proj_hidden, proj_dim, dropout=dropout, last_bn=True)
        self.text_projector = MLP(fused_dim, proj_hidden, proj_dim, dropout=dropout, last_bn=True)
        self.image_predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )
        self.text_predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )
        self.separate_quality_projection = bool(
            pcfg.get("separate_quality_projection", False)
        )
        if self.separate_quality_projection:
            self.quality_image_projector = MLP(
                fused_dim, proj_hidden, proj_dim, dropout=dropout, last_bn=True
            )
            self.quality_image_predictor = nn.Sequential(
                nn.Linear(proj_dim, pred_hidden),
                nn.BatchNorm1d(pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(pred_hidden, proj_dim),
            )
        else:
            self.quality_image_projector = None
            self.quality_image_predictor = None

    def _project_image(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = F.normalize(self.image_projector(feature), dim=-1)
        return {"z_proj": z, "p": F.normalize(self.image_predictor(z), dim=-1)}

    def _project_text(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = F.normalize(self.text_projector(feature), dim=-1)
        return {"z_proj": z, "p": F.normalize(self.text_predictor(z), dim=-1)}

    def _project_quality_image(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project images into the optional quality-only SimSiam space."""
        if not self.separate_quality_projection:
            return self._project_image(feature)
        z = F.normalize(self.quality_image_projector(feature), dim=-1)
        return {
            "z_proj": z,
            "p": F.normalize(self.quality_image_predictor(z), dim=-1),
        }

    def _project_align_image(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project an image through the private alignment representation.

        Image-text SimSiam uses the same alignment-specific feature as the
        downstream alignment head. With the adapter disabled this is exactly
        the legacy path, so old configurations remain unchanged.
        """
        align_feature = self.iqa.align_global_adapter(feature)
        return self._project_image(align_feature)

    def project_patch_regions(
        self,
        patch_features: torch.Tensor,
        region_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Pool human-selected patch regions and send them through SimSiam."""
        weights = region_weights.to(patch_features).clamp(0.0, 1.0)
        pooled = (patch_features * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        # Region-level image/text relations belong to the alignment branch.
        return self._project_align_image(pooled)

    def project_aux_texts(self, texts: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """Project aligned/misaligned prompt phrases with the frozen CLIP path."""
        text_feat = self.iqa.backbone.text_encoder.encode_cached(texts, device)
        text_fused = self.iqa.align_text_adapter(self.iqa.text_to_fused(text_feat))
        return self._project_text(text_fused)

    def _forward_three_views(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        image_clean: torch.Tensor,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Route two SafeAug views to SimSiam and all labels to one clean view."""
        score_clean = self.iqa.forward_single(image_clean, prompts=prompts)
        img1 = self.iqa.backbone.encode_image(image1)
        img2 = self.iqa.backbone.encode_image(image2)

        answer_feat = self.iqa.backbone.encode_text(answers, image1.device)
        answer_fused = self.iqa.text_to_fused(answer_feat)
        # Semantic/image-text relations use the private alignment branch.
        img1_p = self._project_align_image(img1["z"])
        img2_p = self._project_align_image(img2["z"])
        clean_p = self._project_align_image(score_clean["z"])
        # Human quality labels in three-view pretraining must supervise the clean
        # view, not either SafeAug view. Reuse the shared image projection when
        # no separate quality projector is configured.
        quality_clean_p = (
            clean_p
            if not self.separate_quality_projection
            else self._project_quality_image(score_clean["z"])
        )
        if self.separate_quality_projection:
            quality_img1_p = self._project_quality_image(img1["z"])
            quality_img2_p = self._project_quality_image(img2["z"])
        else:
            quality_img1_p = img1_p
            quality_img2_p = img2_p
        prompt_p = self._project_text(score_clean["align_text_feature"])
        align_prompt_p = self._project_text(score_clean["align_text_feature"])
        answer_p = self._project_text(answer_fused)

        out = {
            "img1_z": img1_p["z_proj"],
            "img1_p": img1_p["p"],
            "img2_z": img2_p["z_proj"],
            "img2_p": img2_p["p"],
            "quality_img1_z": quality_img1_p["z_proj"],
            "quality_img1_p": quality_img1_p["p"],
            "quality_img2_z": quality_img2_p["z_proj"],
            "quality_img2_p": quality_img2_p["p"],
            # Clean-view quality projection: use this for all human quality
            # labels (e.g. Artifact/Aesthetics component anchors).
            "quality_clean_z": quality_clean_p["z_proj"],
            "quality_clean_p": quality_clean_p["p"],
            "clean_z": clean_p["z_proj"],
            "clean_p": clean_p["p"],
            "prompt_z": prompt_p["z_proj"],
            "prompt_p": prompt_p["p"],
            "align_prompt_z": align_prompt_p["z_proj"],
            "align_prompt_p": align_prompt_p["p"],
            "answer_z": answer_p["z_proj"],
            "answer_p": answer_p["p"],
            "quality_pred": score_clean["s_quality"],
            "align_pred": score_clean["s_align"],
            "final_pred": score_clean["s_final"],
            "quality_pred_clean": score_clean["s_quality"],
            "align_pred_clean": score_clean["s_align"],
            "final_pred_clean": score_clean["s_final"],
            "align_patch_features_clean": score_clean["align_patch_features"],
        }
        if self.iqa.align_text_bank is not None:
            out["align_text_logits"] = score_clean["align_text_logits"]
            out["align_text_score"] = score_clean["align_text_score"]
        if self.iqa.quality_text_bank is not None:
            out.update(self.iqa.quality_text_bank(out["quality_clean_z"]))
        for component in self.iqa.quality_auxiliary_components:
            key = f"quality_aux_{component}_pred"
            if key in score_clean:
                out[key] = score_clean[key]
                out[f"{key}_clean"] = score_clean[key]
        return out

    def forward_views(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        prompts: Sequence[str],
        answers: Sequence[str],
        image_clean: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if image_clean is not None:
            return self._forward_three_views(
                image1, image2, image_clean, prompts, answers
            )
        # The first call computes prompt features and all persistent IQA heads.
        score1 = self.iqa.forward_single(image1, prompts=prompts)

        # Reuse the same prompt representation for view 2 instead of encoding
        # identical text with frozen CLIP a second time.
        img2 = self.iqa.backbone.encode_image(image2)
        quality_text_fused = (
            score1["text_fused"]
            if self.iqa.quality_branch_use_text_gate
            else None
        )
        q2 = self.iqa._quality_from_patches(
            image2,
            img2["z"],
            img2["f_loc"],
            quality_text_fused,
        )
        a2 = self.iqa._align_from_global_and_patch(
            img2["z"],
            q2["patch_tokens"],
            score1["text_fused"],
            patch_hw=q2.get("patch_hw"),
            text_tokens=score1.get("text_tokens"),
            text_token_ids=score1.get("text_token_ids"),
        )
        fusion_w, _ = self.iqa._compute_fusion_weight(
            img2["z"], score1["text_fused"], q2["s_quality"], a2["s_align"]
        )
        final2 = fusion_w * q2["s_quality"] + (1.0 - fusion_w) * a2["s_align"]

        answer_feat = self.iqa.backbone.encode_text(answers, image1.device)
        answer_fused = self.iqa.text_to_fused(answer_feat)
        # Semantic/image-text relations use the private alignment branch.
        img1_p = self._project_align_image(score1["z"])
        img2_p = self._project_align_image(img2["z"])
        # Semantic projections are paired with Prompt/Answer. Quality
        # projections are paired only with quality-safe views and quality
        # relations. Both receive gradients through the same image encoder.
        if self.separate_quality_projection:
            quality_img1_p = self._project_quality_image(score1["z"])
            quality_img2_p = self._project_quality_image(img2["z"])
        else:
            quality_img1_p = img1_p
            quality_img2_p = img2_p
        prompt_p = self._project_text(score1["align_text_feature"])
        align_prompt_p = self._project_text(score1["align_text_feature"])
        answer_p = self._project_text(answer_fused)

        out = {
            "img1_z": img1_p["z_proj"],
            "img1_p": img1_p["p"],
            "img2_z": img2_p["z_proj"],
            "img2_p": img2_p["p"],
            "quality_img1_z": quality_img1_p["z_proj"],
            "quality_img1_p": quality_img1_p["p"],
            "quality_img2_z": quality_img2_p["z_proj"],
            "quality_img2_p": quality_img2_p["p"],
            "prompt_z": prompt_p["z_proj"],
            "prompt_p": prompt_p["p"],
            "align_prompt_z": align_prompt_p["z_proj"],
            "align_prompt_p": align_prompt_p["p"],
            "answer_z": answer_p["z_proj"],
            "answer_p": answer_p["p"],
            "quality_pred": 0.5 * (score1["s_quality"] + q2["s_quality"]),
            "align_pred": 0.5 * (score1["s_align"] + a2["s_align"]),
            "final_pred": 0.5 * (score1["s_final"] + final2),
            # Per-view predictions let training route human labels only to the
            # complete view while retaining the augmented view for SimSiam.
            "quality_pred1": score1["s_quality"],
            "align_pred1": score1["s_align"],
            "final_pred1": score1["s_final"],
            "quality_pred2": q2["s_quality"],
            "align_pred2": a2["s_align"],
            "final_pred2": final2,
            "align_patch_features1": score1["align_patch_features"],
            "align_patch_features2": a2["align_patch_features"],
        }
        for component in self.iqa.quality_auxiliary_components:
            key = f"quality_aux_{component}_pred"
            if key in score1 and key in q2:
                out[key] = 0.5 * (score1[key] + q2[key])
                out[f"{key}1"] = score1[key]
                out[f"{key}2"] = q2[key]
        return out


def negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return -(p * z).sum(dim=-1)


def weighted_mean(loss_vec: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight is None:
        return loss_vec.mean()
    weight = weight.view(-1).to(loss_vec.device).float()
    return (loss_vec * weight).sum() / (weight.sum() + 1e-6)


def simsiam_pair_loss(p_a: torch.Tensor, z_b: torch.Tensor, p_b: torch.Tensor, z_a: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss_vec = 0.5 * (negative_cosine(p_a, z_b) + negative_cosine(p_b, z_a))
    return weighted_mean(loss_vec, weight)


def load_pretrained_backbone(
    model: CrossModalIQAFullModel,
    checkpoint_path: str,
    strict: bool = False,
    load_scope: str = "auto",
    skip_quality_adapter: bool = False,
) -> None:
    """Load a legacy backbone or a joint-pretraining IQA checkpoint.

    Legacy checkpoints still load only matching ``backbone.*`` keys. Joint
    checkpoints store the persistent downstream model under ``iqa.*``; those
    keys can be loaded fully or selectively. SimSiam-only projector/predictor
    keys are ignored, except that an enabled downstream ``quality_adapter``
    imports ``quality_image_projector``. The predictor remains disposable.

    skip_quality_adapter leaves the existing Adapter at its seeded initialization,
    including BN buffers, without disabling or replacing the module.

    Scopes:
      - auto/full: legacy behavior; load all compatible joint-IQA keys.
      - representation: load shared visual/text projections plus private
        quality/alignment adapters.
      - backbone: load ViT, ResNet and visual fusion only.
    """
    load_scope = str(load_scope or "auto").strip().lower()
    if load_scope not in {"auto", "full", "representation", "backbone"}:
        raise ValueError(
            f"Unknown pretrained load scope: {load_scope}. "
            "Choose from: auto, full, representation, backbone."
        )

    representation_prefixes = (
        "backbone.global_encoder.",
        "backbone.local_encoder.",
        "backbone.fusion.",
        "backbone.text_encoder.context_vectors",
        "text_to_fused.",
        "quality_adapter.",
        "quality_text_bank.",
        "align_text_bank.",
        "align_global_adapter.",
        "align_text_adapter.",
        "align_patch_adapter.",
        "align_vmamba.",
    )
    backbone_prefixes = representation_prefixes[:3]

    # ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_device = next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=model_device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    target = model.state_dict()
    # In the decoupled checkpoint, the quality SimSiam projector is the
    # quality representation that should survive downstream. Prefer it over
    # the auxiliary persistent IQA adapter when both have the same shape.
    has_quality_simsiam_projector = any(
        (key[7:] if key.startswith("module.") else key).startswith(
            "quality_image_projector."
        )
        for key in state
    )
    loadable = {}
    for k, v in state.items():
        source_key = k[7:] if k.startswith("module.") else k
        if skip_quality_adapter and source_key.startswith((
            "iqa.quality_adapter.", "quality_adapter.", "quality_image_projector.",
        )):
            continue
        if (
            has_quality_simsiam_projector
            and source_key.startswith("iqa.quality_adapter.")
        ):
            continue
        if source_key.startswith("iqa."):
            target_key = source_key[len("iqa."):]
        elif source_key.startswith("backbone."):
            target_key = source_key
        elif source_key.startswith("quality_adapter."):
            target_key = source_key
        elif (
            getattr(model, "use_quality_adapter", False)
            and source_key.startswith("quality_image_projector.")
        ):
            target_key = "quality_adapter." + source_key[len("quality_image_projector."):]
        else:
            continue
        if load_scope == "representation" and not target_key.startswith(representation_prefixes):
            continue
        if load_scope == "backbone" and not target_key.startswith(backbone_prefixes):
            continue
        if (
            target_key in target
            and target_key.startswith("backbone.global_encoder.")
            and target[target_key].shape != v.shape
        ):
            raise RuntimeError(
                f"CLIP checkpoint architecture mismatch at {target_key}: "
                f"checkpoint={tuple(v.shape)}, model={tuple(target[target_key].shape)}. "
                "Use --clip-large for both pretraining and downstream, or neither."
            )
        if target_key in target and target[target_key].shape == v.shape:
            loadable[target_key] = v
    context = target.get("backbone.text_encoder.context_vectors")
    if context is not None and load_scope != "backbone" and (
        "backbone.text_encoder.context_vectors" not in loadable
    ):
        raise RuntimeError(
            "Learnable CLIP text context is enabled, but the checkpoint has no "
            "compatible context_vectors. Use its matching pretraining checkpoint "
            "and the same --text-context-tokens value."
        )
    if not skip_quality_adapter and getattr(model, "use_quality_adapter", False) and not any(
        key.startswith("quality_adapter.") for key in loadable
    ):
        raise RuntimeError(
            "model.quality_adapter.enabled=true, but the checkpoint did not provide "
            "compatible quality_image_projector/quality_adapter weights. Use the "
            "dual-projection checkpoint with pretrained_load_scope=representation or full."
        )
    if getattr(model, "use_quality_text_bank", False) and load_scope != "backbone":
        required_bank_keys = {
            "quality_text_bank.text_features",
            "quality_text_bank.centers",
            "quality_text_bank.image_to_text.weight",
        }
        if getattr(model.quality_text_bank, "class_context", None) is not None:
            required_bank_keys.add("quality_text_bank.class_context")
        missing_bank_keys = required_bank_keys.difference(loadable)
        if missing_bank_keys:
            raise RuntimeError(
                "Quality text bank is enabled but the checkpoint lacks compatible "
                f"weights/buffers: {sorted(missing_bank_keys)}"
            )
        if not torch.allclose(
            loadable["quality_text_bank.centers"],
            model.quality_text_bank.centers,
            atol=1e-6,
        ):
            raise RuntimeError("Quality text centers differ from the pretraining checkpoint.")
    if getattr(model, "use_align_text_bank", False) and load_scope != "backbone":
        required_align_bank_keys = {
            "align_text_bank.text_features",
            "align_text_bank.centers",
            "align_text_bank.image_to_text.weight",
        }
        if getattr(model.align_text_bank, "class_context", None) is not None:
            required_align_bank_keys.add("align_text_bank.class_context")
        missing_align_bank_keys = required_align_bank_keys.difference(loadable)
        if missing_align_bank_keys:
            raise RuntimeError(
                "Align text bank is enabled but the checkpoint lacks compatible "
                f"weights/buffers: {sorted(missing_align_bank_keys)}"
            )
        if not torch.allclose(
            loadable["align_text_bank.centers"],
            model.align_text_bank.centers,
            atol=1e-6,
        ):
            raise RuntimeError("Align text centers differ from the pretraining checkpoint.")
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    # Older class-context checkpoints predate the initialization anchor buffer.
    # Treat their loaded context as the anchor so enabling the new regularizer
    # never pulls an old checkpoint toward a fresh random initialization.
    for bank_name in ("quality_text_bank", "align_text_bank"):
        bank = getattr(model, bank_name, None)
        context_key = f"{bank_name}.class_context"
        initial_key = f"{bank_name}.initial_class_context"
        if bank is not None and context_key in loadable and initial_key not in loadable:
            initial = getattr(bank, "initial_class_context", None)
            context = getattr(bank, "class_context", None)
            if initial is not None and context is not None and initial.shape == context.shape:
                initial.copy_(context.detach())
    visual_count = sum(k.startswith(backbone_prefixes) for k in loadable)
    semantic_count = sum(k.startswith("text_to_fused.") for k in loadable)
    text_context_count = int("backbone.text_encoder.context_vectors" in loadable)
    quality_adapter_count = sum(k.startswith("quality_adapter.") for k in loadable)
    task_count = len(loadable) - visual_count - semantic_count - text_context_count - quality_adapter_count
    print(
        f"[INFO] Loaded {len(loadable)} compatible keys from {checkpoint_path} "
        f"(scope={load_scope}, visual={visual_count}, "
        f"text_to_fused={semantic_count}, text_context={text_context_count}, "
        f"quality_adapter={quality_adapter_count}, "
        f"task={task_count})"
    )
    if skip_quality_adapter:
        print("[INFO] Quality Adapter checkpoint weights and BN buffers skipped; seeded random Adapter retained.")
    if strict and (missing or unexpected):
        print("[WARN] missing:", missing)
        print("[WARN] unexpected:", unexpected)
