"""Opt-in 40 joint + 20 frozen-local epochs on one uninterrupted 60-epoch run."""
import math
from pathlib import Path
import torch
from utils.protected_local import ALLOWED,frozen_digest,write_json_new

VERSION='align_40plus20_v1'
OPTION={'version':VERSION,'freeze_start_epoch':41}
PREFIXES=ALLOWED['scratch']


def enabled(cfg):return bool(cfg.get('train',{}).get('fair_40plus20'))


def validate(cfg):
    t=cfg.get('train',{});option=t.get('fair_40plus20')
    if option is None:return
    m=cfg.get('model',{})
    if (option!=OPTION or t.get('epochs')!=60 or not t.get('compare_selection_policies')
            or t.get('selection_comparison_version')!='quality_mean_v1'
            or t.get('progressive_unfreeze',{}).get('enabled') or t.get('refit_plan')
            or t.get('save_selected_downstream') or t.get('protected_local_variant')
            or t.get('pcgrad',{}).get('enabled') or t.get('use_joint_iqa_pretraining')
            or m.get('fusion',{}).get('mode')!='projection_only_v1'
            or m.get('align_region_word')!={'version':'clip_region_word_v1'}):
        raise ValueError('40+20 requires one 60-epoch region-word run, dual validation selection, no refit/progressive/restart')


def step_counts(model,optimizer):
    return {name:int(optimizer.state.get(parameter,{}).get('step',0))
            for name,parameter in model.named_parameters() if name.startswith(PREFIXES)}


def group_lrs(optimizer):return {g['name']:float(g['lr']) for g in optimizer.param_groups}


def enforce_local_eval(model,cfg,epoch):
    if not enabled(cfg) or epoch<=40:return
    if not hasattr(model,'_fair_transition'):raise RuntimeError('Missing 40+20 phase transition')
    # model.train() was called by the ordinary trainer. Re-freeze stochastic
    # modes as well as weights, without reconstructing the model or optimizer.
    model.eval();model.training=True
    for prefix in PREFIXES:model.get_submodule(prefix.rstrip('.')).train(True)


def begin_epoch(model,optimizer,cfg,epoch):
    if not enabled(cfg):return None
    validate(cfg)
    if epoch<=40:return None
    if not hasattr(model,'_fair_transition'):
        if epoch!=41 or not hasattr(model,'_fair_boundary_validation'):
            raise RuntimeError('40+20 must continue directly from the epoch-40 current state')
        if not isinstance(optimizer,torch.optim.AdamW):raise ValueError('40+20 requires baseline AdamW')
        before_steps=step_counts(model,optimizer)
        if not before_steps or any(n<=0 for n in before_steps.values()):
            raise RuntimeError('Local optimizer state was not accumulated during the first 40 epochs')
        model._fair_optimizer_identity=id(optimizer)
        model.protected_prefixes=PREFIXES
        model._fair_transition=dict(version=VERSION,boundary_epoch=40,start_epoch=41,
            frozen_before=frozen_digest(model),boundary_validation=model._fair_boundary_validation,
            local_steps_before=before_steps,lrs_at_transition=group_lrs(optimizer),
            optimizer='AdamW',optimizer_recreated=False,scheduler_restarted=False,
            warmup_used=False,source_state='current_epoch40_not_validation_selected')
    if model._fair_optimizer_identity!=id(optimizer):raise RuntimeError('40+20 optimizer was replaced')
    # Existing progressive-stage bookkeeping restores original requires_grad
    # each epoch even when disabled; explicitly enforce the freeze afterwards.
    for name,parameter in model.named_parameters():
        active=name.startswith(PREFIXES)
        parameter.requires_grad_(active)
        if not active:parameter.grad=None
    return model._fair_transition


def frozen_metrics(metrics):
    return {k:v for k,v in metrics.items() if k.startswith('val_quality_')
            or k in ('val_align_global_SROCC','val_align_global_PLCC')}


def record_validation(model,cfg,epoch,metrics):
    if not enabled(cfg):return
    if epoch==40:
        model._fair_boundary_validation=dict(frozen_metrics(metrics))
        model._fair_local_validations=[]
    elif epoch>40:
        if not hasattr(model,'_fair_transition'):raise RuntimeError('Missing frozen phase evidence')
        reference=model._fair_transition['boundary_validation']
        for key,value in reference.items():
            if not math.isclose(float(metrics[key]),float(value),rel_tol=0,abs_tol=1e-6):
                raise RuntimeError('Frozen Quality/global validation metric changed: '+key)
        model._fair_local_validations.append(epoch)


def finalize(model,optimizer,scheduler,cfg):
    """Must run BEFORE selection tracker restores any earlier-epoch model."""
    if not enabled(cfg):return None
    if (not hasattr(model,'_fair_transition') or model._fair_local_validations!=list(range(41,61))
            or scheduler.T_max!=60 or scheduler.last_epoch!=60):
        raise RuntimeError('Incomplete 40+20 history or noncontinuous cosine schedule')
    proof=dict(model._fair_transition)
    proof.update(last_epoch=60,joint_epochs=40,local_epochs=20,
        frozen_after=frozen_digest(model),local_steps_after=step_counts(model,optimizer),
        lrs_final=group_lrs(optimizer),scheduler_T_max=scheduler.T_max,scheduler_last_epoch=scheduler.last_epoch,
        trainable_parameters=[name for name,p in model.named_parameters() if p.requires_grad])
    if proof['frozen_before']!=proof['frozen_after']:raise RuntimeError('Frozen parameters/buffers changed')
    if any(proof['local_steps_after'][key]<=value for key,value in proof['local_steps_before'].items()):
        raise RuntimeError('Local optimizer state did not continue through phase two')
    write_json_new(Path(cfg['output_dir'])/'fair_phase_manifest.json',proof)
    return proof
