"""Versioned full downstream snapshots and explicit frozen local-branch adaptation."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from models.two_stage_models import CrossModalIQAFullModel
from utils.selection_policies import cpu_state_dict
from utils.run_metadata import checkpoint_fingerprint

FULL_FORMAT='selected_downstream_v1'
LOCAL_FORMAT='protected_local_v1'
ALLOWED={
    'old':('align_patch_adapter.','align_patch_img_proj.','align_patch_txt_proj.','align_patch_head.','align_gate.'),
    'scratch':('align_region_word.','align_patch_head.','align_gate.'),
    'warm':('align_region_word.','align_patch_head.','align_gate.'),
    'warmup':('align_region_word.','align_patch_head.'),
}


def sha256(path):
    return checkpoint_fingerprint(path)['sha256']


def write_json_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def validate_snapshot_config(cfg):
    version=cfg.get('train',{}).get('save_selected_downstream')
    if version is None:return
    if (version!=FULL_FORMAT or cfg['model']['fusion'].get('mode')!='projection_only_v1'
            or cfg['train'].get('compare_selection_policies') or cfg['train'].get('refit_plan')
            or cfg['train'].get('selection_metric')!='val_quality_SROCC'
            or cfg['train'].get('selection_mode')!='max'):
        raise ValueError('Full snapshot requires projection_only_v1 single Quality selection; no refit/comparison')


def nonpersistent_buffers(model):
    persisted=set(model.state_dict())
    return {n:b for n,b in model.named_buffers() if n not in persisted}


def save_selected_downstream(model,cfg,final,upstream):
    if cfg.get('train',{}).get('save_selected_downstream') is None:return None
    validate_snapshot_config(cfg)
    if (final['last_epoch']!=cfg['train']['epochs'] or final['selection_metric']!='val_quality_SROCC'
            or not 1<=final['best_epoch']<=final['last_epoch'] or not upstream):
        raise ValueError('Cannot save an unfinished or incorrectly selected foundation')
    path=Path(cfg['output_dir'])/'best_selected_downstream.pth'
    checkpoint=dict(format=FULL_FORMAT,training_complete=True,mode='projection_only_v1',
        run_id=cfg['run_id'],cfg=copy.deepcopy(cfg),upstream=upstream,
        epoch=final['best_epoch'],last_epoch=final['last_epoch'],selection_metric=final['selection_metric'],
        best_val=final['best_val'],best_test=final['best_test'],model=cpu_state_dict(model),
        nonpersistent_buffers={n:b.detach().cpu().clone() for n,b in nonpersistent_buffers(model).items()})
    with path.open('xb') as stream:torch.save(checkpoint,stream)
    record=dict(format=FULL_FORMAT,checkpoint=path.name,sha256=sha256(path),
        run_id=cfg['run_id'],selected_epoch=checkpoint['epoch'],last_epoch=checkpoint['last_epoch'],
        upstream_sha256=upstream['sha256'])
    write_json_new(path.parent/'selected_checkpoint_manifest.json',record)
    return record


def restore_foundation(model,checkpoint,cfg,expected_run_id=None,expected_pretrain_sha=None):
    if (checkpoint.get('format')!=FULL_FORMAT or checkpoint.get('training_complete') is not True
            or checkpoint.get('mode')!='projection_only_v1' or checkpoint.get('last_epoch')!=60
            or checkpoint.get('selection_metric')!='val_quality_SROCC'):
        raise ValueError('Requires a completed 60-epoch full downstream foundation')
    if expected_run_id is not None and checkpoint.get('run_id')!=expected_run_id:
        raise ValueError('Wrong foundation run identity')
    if expected_pretrain_sha is not None and checkpoint.get('upstream',{}).get('sha256')!=expected_pretrain_sha:
        raise ValueError('Wrong original pretraining source')
    original=checkpoint['cfg']
    if (any(original[k]!=cfg[k] for k in ('seed','split_seed'))
            or original['dataset']!=cfg['dataset']):
        raise ValueError('Foundation dataset/seed/split mismatch')
    target=model.state_dict();source=checkpoint['model']
    new={k for k in target if k.startswith('align_region_word.')}
    if set(source)!=set(target)-new:
        raise ValueError('Incomplete/foreign foundation state; only new region projections may be absent')
    if any(source[k].shape!=target[k].shape for k in source):raise ValueError('Foundation tensor shape mismatch')
    source_buffers=checkpoint.get('nonpersistent_buffers')
    target_buffers=nonpersistent_buffers(model)
    if source_buffers is None or set(source_buffers)!=set(target_buffers):
        raise ValueError('Incomplete full foundation nonpersistent buffers')
    with torch.no_grad():
        for name,buffer in target_buffers.items():
            value=source_buffers[name]
            if buffer.shape!=value.shape or buffer.dtype!=value.dtype:
                raise ValueError('Foundation nonpersistent buffer shape/dtype mismatch')
            buffer.copy_(value.to(buffer.device))
    missing,unexpected=model.load_state_dict(source,strict=False)
    if set(missing)!=new or unexpected:raise ValueError('Unexpected foundation load difference')
    return dict(inherited_tensors=len(source),new_parameters=sorted(new),
                scoring_heads='fully_inherited_from_selected_downstream')


class ProtectedLocalIQA(CrossModalIQAFullModel):
    def protect(self,variant):
        if variant not in ALLOWED:raise ValueError('Unknown protected local variant')
        if (variant=='old') != (self.align_region_word is None):
            raise ValueError('Local variant/model route mismatch')
        self.protected_variant=variant
        self.protected_prefixes=ALLOWED[variant]
        self.requires_grad_(False)
        for name in self.protected_prefixes:
            self.get_submodule(name.rstrip('.')).requires_grad_(True)
        self.train(True)

    def train(self,mode=True):
        if not hasattr(self,'protected_prefixes'):return super().train(mode)
        # Frozen BN statistics AND frozen Dropout must remain in eval. Only
        # explicitly listed local modules may train; root flag is informational.
        super().train(False)
        self.training=mode
        for name in self.protected_prefixes:self.get_submodule(name.rstrip('.')).train(mode)
        return self

    def forward_single(self,*args,**kwargs):
        out=super().forward_single(*args,**kwargs)
        if getattr(self,'protected_variant',None)=='warmup':
            # RichHF warmup targets the new local path alone, never the random
            # global scoring head or the mixed alignment prediction.
            out['s_align']=out['s_align_patch']
        return out


def frozen_digest(model):
    allowed=model.protected_prefixes
    h=hashlib.sha256()
    state=dict(model.state_dict());state.update(nonpersistent_buffers(model))
    for name,tensor in sorted(state.items()):
        if name.startswith(allowed):continue
        h.update((name+str(tensor.dtype)+str(tuple(tensor.shape))).encode())
        data=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        h.update(data.tobytes())
    return h.hexdigest()


def transfer_warm_projection(model,checkpoint,expected_source_sha):
    if (checkpoint.get('format')!=LOCAL_FORMAT or checkpoint.get('variant')!='warmup' or checkpoint.get('mode')!='projection_only_v1'
            or not checkpoint.get('training_complete')
            or checkpoint.get('upstream',{}).get('sha256')!=expected_source_sha
            or not checkpoint.get('frozen_before')
            or checkpoint.get('frozen_before')!=checkpoint.get('frozen_after')):
        raise ValueError('Wrong/incomplete local pretraining source')
    # This is an intentional new-dataset transfer policy: ONLY two projection
    # tensors, never the RichHF score head or any global/quality parameter.
    prefix='align_region_word.'
    source={k.removeprefix(prefix):v for k,v in checkpoint['model'].items() if k.startswith(prefix)}
    model.align_region_word.load_state_dict(source,strict=True)
    return dict(loaded=sorted(prefix+k for k in source),
        explicitly_not_transferred=sorted(k for k in checkpoint['model'] if not k.startswith(prefix)),
        local_scoring_head='retained_from_mature_downstream_foundation')
