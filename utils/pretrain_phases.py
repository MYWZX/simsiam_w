"""Opt-in quality-first pretraining; default joint training is unchanged."""
import copy
import hashlib
import json
from pathlib import Path

import torch

QUALITY_PREFIXES = (
    'iqa.backbone.global_encoder.', 'iqa.backbone.local_encoder.',
    'iqa.backbone.fusion.semantic_proj.', 'iqa.backbone.global_adapter.',
    'iqa.local_patch_proj.', 'iqa.nss_proj.', 'iqa.nss_gate.',
    'iqa.quality_adapter.', 'iqa.quality_adapter_gate_logit',
    'iqa.quality_local_encoder.', 'iqa.quality_feature_mix_logit',
    'iqa.quality_mix_logit', 'iqa.global_quality_head.', 'iqa.patch_quality_head.',
    'iqa.quality_aux_heads.', 'quality_image_predictor.',
)
# I-I belongs to quality stage, so PI/image Predictor are shared across stages.
QUALITY_RELATION_PREFIXES = ('iqa.shared_image_projector.', 'image_predictor.')
ALIGN_PREFIXES = (
    'iqa.shared_image_projector.', 'iqa.shared_text_projector.',
    'iqa.text_to_fused.', 'iqa.backbone.text_encoder.context_vectors',
    'iqa.align_global_adapter.', 'iqa.align_patch_adapter.', 'iqa.align_text_adapter.',
    'iqa.align_patch_img_proj.', 'iqa.align_patch_txt_proj.',
    'iqa.align_global_head.', 'iqa.align_patch_head.', 'iqa.align_gate.',
    'image_predictor.', 'text_predictor.',
)
ALIGN_LOSSES = ('lambda_img_answer', 'lambda_img_prompt', 'lambda_prompt_answer',
    'lambda_prompt_answer_gap', 'lambda_align_pretrain_reg', 'lambda_align_pretrain_rank',
    'lambda_align_pretrain_srocc', 'lambda_align_text_ce', 'lambda_img_prompt_local',
    'lambda_prompt_answer_word')
QUALITY_LOSSES = ('lambda_img_siam', 'lambda_quality_view_siam', 'lambda_rank',
    'lambda_pretrain_srocc', 'lambda_quality_reg', 'lambda_quality_rank', 'lambda_quality_srocc',
    'lambda_quality_neighbor_siam', 'lambda_quality_order_siam', 'lambda_quality_triplet_siam',
    'lambda_quality_text_ce', 'lambda_quality_anchor', 'lambda_quality_ordinal',
    'lambda_artifact_anchor', 'lambda_artifact_ordinal', 'lambda_aesthetics_anchor',
    'lambda_aesthetics_ordinal', 'lambda_residual_huber')


def phase(cfg):
    value = cfg.get('train', {}).get('pretraining_phase', 'joint')
    if value not in ('joint', 'quality', 'alignment'):
        raise ValueError('Unknown pretraining_phase: '+str(value))
    return value


def validate(cfg):
    p = phase(cfg)
    if p == 'joint':
        return
    t, m = cfg['train'], cfg['model']
    if (m['fusion'].get('mode') != 'projection_only_v1' or
            not m.get('dual_quality_representation', {}).get('enabled') or
            not m.get('shared_simsiam_projection') or not t.get('two_output_only') or
            t.get('human_supervision_view') != 'three_view'):
        raise ValueError('Quality-first phases require projection_only_v1, dual quality, PI/PT, three_view and two outputs')
    if m.get('quality_branch_use_text_gate', True):
        raise ValueError('Frozen quality must not depend on trainable text context')
    disabled = ALIGN_LOSSES if p == 'quality' else QUALITY_LOSSES
    if any(float(t.get(k, 0)) != 0 for k in disabled):
        raise ValueError('Inactive phase has nonzero loss coefficients')
    if any(float(t.get('lambda_final_pretrain_'+k, 0)) != 0 for k in ('reg','rank','srocc')):
        raise ValueError('Phase isolation forbids final mixed-task losses')
    if p == 'alignment' and t.get('quality_auxiliary_supervision', {}).get('enabled', False):
        raise ValueError('Alignment phase must disable quality auxiliary losses')
    metric = 'val_component_anchor_SROCC' if p == 'quality' else 'val_align_proxy_SROCC'
    if t.get('pretrain_selection_metric') != metric or t.get('pretrain_selection_mode') != 'max':
        raise ValueError('Wrong phase selection rule')
    if p == 'quality' and t.get('quality_stage_checkpoint'):
        raise ValueError('Quality stage must start fresh')
    if p == 'alignment' and not t.get('quality_stage_checkpoint'):
        raise ValueError('Alignment stage needs its completed quality-stage checkpoint')


def configure(model, cfg):
    validate(cfg)
    p = phase(cfg)
    if p == 'joint':
        return
    allowed = QUALITY_PREFIXES + QUALITY_RELATION_PREFIXES if p == 'quality' else ALIGN_PREFIXES
    model._phase_allowed_modules = allowed
    for name, param in model.named_parameters():
        if not name.startswith(allowed):
            param.requires_grad_(False)
    if not any(x.requires_grad for x in model.parameters()):
        raise ValueError('Phase has no trainable parameters')
    enforce_eval(model, cfg)


def active_module(name, allowed):
    # Affine-free BN and Dropout have no parameters but must still train when
    # they belong to an active projector. Parameter-count heuristics are wrong.
    if not name:
        return True
    return any(name == p.rstrip('.') or name.startswith(p.rstrip('.')+'.')
               or p.rstrip('.').startswith(name+'.') for p in allowed)


def enforce_eval(model, cfg):
    p = phase(cfg)
    if p == 'joint':
        return
    allowed = QUALITY_PREFIXES + QUALITY_RELATION_PREFIXES if p == 'quality' else ALIGN_PREFIXES
    for name, module in model.named_modules():
        if not active_module(name, allowed):
            module.eval()


def frozen_digest(model):
    """Hash frozen parameters and buffers in explicitly frozen module paths."""
    allowed = model._phase_allowed_modules
    frozen = {n for n,p in model.named_parameters() if not p.requires_grad}
    for name, module in model.named_modules():
        if not active_module(name, allowed):
            frozen.update((name+'.' if name else '')+n for n,_ in module.named_buffers(recurse=False))
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name in frozen:
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(model, cfg):
    """Strict full model transfer, new optimizer; never a resume or mixed source."""
    validate(cfg)
    if phase(cfg) == 'joint':
        return None
    source = None
    if phase(cfg) == 'alignment':
        from utils.run_metadata import checkpoint_fingerprint
        path = cfg['train']['quality_stage_checkpoint']
        source = checkpoint_fingerprint(path)
        expected = json.loads((Path(cfg['output_dir'])/'quality_stage_source.json').read_text())
        if (source['sha256'] != expected['full_sha256'] or
                Path(source['path']).resolve() != Path(expected['full_checkpoint']).resolve()):
            raise ValueError('Quality source fingerprint differs before loading')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        scfg = checkpoint['cfg']
        validate(scfg)
        if (checkpoint['epoch'] != expected['selected_epoch'] or scfg.get('train', {}).get('pretraining_phase') != 'quality' or
                checkpoint.get('run_id') != cfg['train']['quality_stage_run_id'] or
                checkpoint.get('training_state', {}).get('best_epoch') != checkpoint['epoch']):
            raise ValueError('Expected the matching quality-stage full best checkpoint')
        sm, tm = copy.deepcopy(scfg['model']), copy.deepcopy(cfg['model'])
        for m in (sm, tm):
            for encoder in ('global_encoder', 'local_encoder'):
                m[encoder].pop('freeze', None)
        if sm != tm or scfg['seed'] != cfg['seed'] or scfg.get('split_seed') != cfg.get('split_seed'):
            raise ValueError('Quality-stage architecture/seed mismatch')
        if scfg['dataset'] != cfg['dataset']:
            raise ValueError('Quality-stage dataset/split mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        if checkpoint_fingerprint(path) != source:
            raise ValueError('Quality source changed during loading')
        del checkpoint
    configure(model, cfg)
    return {'phase': phase(cfg), 'source': source, 'strict_full_model_transfer': source is not None,
            'optimizer_reset': True, 'trainable_parameters': [n for n,p in model.named_parameters() if p.requires_grad],
            'frozen_sha256_before': frozen_digest(model)}


def finish(model, record, output_dir):
    if record is None:
        return
    after = frozen_digest(model)
    if after != record['frozen_sha256_before']:
        raise RuntimeError('Frozen phase parameters or buffers changed; completion refused')
    record = dict(record, frozen_sha256_after=after, frozen_state_unchanged=True)
    with (Path(output_dir)/'phase_manifest.json').open('x') as f:
        json.dump(record, f, indent=2)
