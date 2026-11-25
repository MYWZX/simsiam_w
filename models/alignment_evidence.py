"""Opt-in evidence diagnostics, bounded temperature and native CLIP reference.

Native CLIP means the CURRENT paired 768-D encoder space, with the same five
contexts. It is not a second frozen original CLIP teacher or a new weight source.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def validate_evidence(config, region=False):
    allowed = {'version','diagnostics','temperature','coverage','native_reference'}
    if set(config)-allowed or config.get('version') != 'align_evidence_v1':
        raise ValueError('Unknown align_evidence version/options')
    temp = config.get('temperature', {})
    if set(temp)-{'mode','initial','minimum','maximum'}:
        raise ValueError('Unknown temperature options')
    mode = temp.get('mode', 'fixed')
    lo, hi, initial = (float(temp.get(k,v)) for k,v in
                       [('minimum',.03),('maximum',2.),('initial',1.)])
    if mode not in ('fixed','learned') or not all(math.isfinite(x) for x in (lo,hi,initial)) or not 0<lo<initial<hi:
        raise ValueError('Temperature requires 0 < minimum < initial < maximum')
    cov = config.get('coverage', {})
    if set(cov)-{'mode','temperature','lexical_mask'} or cov.get('mode','mean') not in ('mean','softmin'):
        raise ValueError('Unknown coverage options')
    ct = float(cov.get('temperature',.1))
    if not math.isfinite(ct) or ct<=0 or (cov.get('mode','mean')!='mean' and not region):
        raise ValueError('Softmin coverage requires region-token matching and positive temperature')
    ref = config.get('native_reference', {})
    if set(ref)-{'enabled','initial_mix'}:
        raise ValueError('Unknown native reference options')
    mix = float(ref.get('initial_mix',.05))
    if not math.isfinite(mix) or not 0<mix<1:
        raise ValueError('Reference mix must be in (0,1)')
    return lo,hi,initial,mode


class AlignmentEvidence(nn.Module):
    def __init__(self, config, region=False):
        super().__init__()
        lo,hi,initial,mode = validate_evidence(config,region)
        self.config = config
        self.minimum, self.maximum, self.initial = lo,hi,initial
        self.diagnostics = bool(config.get('diagnostics',True))
        if mode == 'learned':
            self.temperature_logit = nn.Parameter(torch.tensor(math.log((initial-lo)/(hi-initial))))
        else:
            self.register_parameter('temperature_logit',None)
        self.native_reference = bool(config.get('native_reference',{}).get('enabled',False))
        if self.native_reference:
            self.clip_scale = nn.Parameter(torch.tensor(1.))
            self.clip_bias = nn.Parameter(torch.tensor(0.))
            mix = float(config.get('native_reference',{}).get('initial_mix',.05))
            self.clip_mix_logit = nn.Parameter(torch.tensor(math.log(mix/(1-mix))))

    def temperature(self):
        if self.temperature_logit is None:
            return self.initial
        return self.minimum+(self.maximum-self.minimum)*self.temperature_logit.sigmoid()

    def reference(self, image, text, score):
        if image.shape != text.shape:
            raise ValueError('Native CLIP reference requires matched projected encoder dimensions')
        cosine = (F.normalize(image.float(),dim=-1)*F.normalize(text.float(),dim=-1)).sum(-1)
        calibrated = torch.sigmoid(self.clip_scale*cosine+self.clip_bias)
        mix = self.clip_mix_logit.sigmoid()
        return (1-mix)*score+mix*calibrated, {'native_cosine':cosine,
                'native_score':calibrated, 'native_mix':mix.expand_as(score)}


def spatial_diagnostics(weights):
    """Per-image normalized entropy and max, averaged over valid text columns."""
    p = weights.detach().float()
    if p.ndim == 2:
        p = p.unsqueeze(-1)
    valid = p.sum(1)>0
    count = valid.sum(-1).clamp_min(1)
    entropy = -(p*p.clamp_min(1e-12).log()).sum(1)/max(math.log(p.size(1)),1e-12)
    return {'spatial_entropy':(entropy*valid).sum(-1)/count,
            'spatial_max':(p.max(1).values*valid).sum(-1)/count}


def token_coverage(image, text, weights, mask, config):
    per_token = weights.transpose(1,2) @ image
    match = (F.normalize(per_token,dim=-1)*text).sum(-1)
    if config.get('mode','mean') == 'softmin':
        logits = (-match/float(config.get('temperature',.1))).masked_fill(~mask,-torch.inf)
        importance = logits.softmax(-1)
    else:
        importance = mask/mask.sum(-1,keepdim=True)
    evidence = (per_token*importance[...,None]).sum(1)
    local_text = (text*importance[...,None]).sum(1)
    return evidence,local_text,match,importance


def lexical_mask(ids, mask, tokenizer):
    """Drop punctuation-only BPE pieces; retain lexical subwords, not word labels.

    No stop-word/POS/parser claims. For punctuation-only prompts, fall back to
    the existing content mask and report that fallback instead of producing NaN.
    """
    unique = ids[mask].unique().detach().cpu().tolist()
    lexical = {i:any(c.isalnum() for c in tokenizer.decode([i])) for i in unique}
    keep = torch.zeros_like(mask)
    for i,valid in lexical.items():
        if valid:
            keep |= ids.eq(i)
    result = mask & keep
    fallback = ~result.any(-1)
    return torch.where(fallback[:,None],mask,result),fallback


def truncation_flags(encoder, prompts, device):
    # Full-budget tokenization is enough to detect overflow of the smaller
    # learned-context budget, including strings longer than full CLIP capacity.
    full = encoder.tokenizer([str(p) for p in prompts]).to(device)
    length = full.argmax(-1)-1
    budget = int(encoder.model.context_length)-int(encoder.num_context_tokens)-2
    return (length>budget).float()
