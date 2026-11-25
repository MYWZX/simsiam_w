"""Downstream-only cosine region/BPE-token matching; no cross-attention blocks."""
import torch
from torch import nn
from torch.nn import functional as F


def content_token_mask(ids, num_context_tokens):
    """Standard CLIP tokenizer: EOT is highest ID. Exclude SOT/context/EOT/pad.

    Content is BPE subtokens, not a claim of word-level annotation or labels.
    ID zero can be a genuine content token, so do not mask by ids != 0.
    """
    positions = torch.arange(ids.size(1), device=ids.device)[None, :]
    mask = (positions > num_context_tokens) & (positions < ids.argmax(-1, keepdim=True))
    if not bool(mask.any(-1).all()):
        raise ValueError("Region-word matching requires a nonempty Prompt")
    return mask


class RegionWordAlignment(nn.Module):
    def __init__(self, visual_width, text_width, dim):
        super().__init__()
        self.image_projection = nn.Linear(visual_width, dim, bias=False)
        self.text_projection = nn.Linear(text_width, dim, bias=False)

    def forward(self, regions, tokens, mask, temperature=None, coverage=None):
        image = F.normalize(self.image_projection(regions), dim=-1, eps=1e-6)
        text = F.normalize(self.text_projection(tokens), dim=-1, eps=1e-6)
        # Every content token retrieves evidence over image regions. Mask AFTER
        # spatial softmax; masked columns cannot affect any valid column.
        logits = image @ text.transpose(1, 2)
        if temperature is not None:
            logits = logits/temperature
        weights = torch.softmax(logits, dim=1) * mask[:, None, :]
        if coverage is not None:
            from models.alignment_evidence import token_coverage
            evidence, local_text, match, importance = token_coverage(image,text,weights,mask,coverage)
            return evidence,local_text,weights,match,importance
        per_token = weights.transpose(1, 2) @ image
        count = mask.sum(-1, keepdim=True).to(image.dtype)
        evidence = (per_token * mask[..., None]).sum(1) / count
        text_mean = (text * mask[..., None]).sum(1) / count
        return evidence, text_mean, weights
