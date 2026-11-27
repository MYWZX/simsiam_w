"""Opt-in complete selected-model snapshots for read-only semantic diagnostics.

No optimizer/resume support. Historical log-only experiments cannot be recovered
from this format. Default training behavior and files remain unchanged.
"""
import copy
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch
FORMAT='semantic_selected_v1'
FLAG='export_semantic_snapshots'


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=False).encode()).hexdigest()


def enabled(cfg):
    flag=cfg.get('train',{}).get(FLAG)
    if flag is None:return False
    t=cfg['train']
    if (flag!=FORMAT or t.get('epochs')!=60 or t.get('refit_plan') or t.get('quality_only')
            or cfg['model']['fusion'].get('mode')!='projection_only_v1'
            or t.get('selection_metric')!='val_quality_SROCC' or t.get('selection_mode')!='max'
            or (t.get('compare_selection_policies') and t.get('selection_comparison_version')!='quality_mean_v1')):
        raise ValueError('Semantic export requires completed-budget projection_only_v1 Quality/dual selection, no refit')
    return True


def buffers(model):
    persistent=set(model.state_dict())
    return {n:v.detach().to('cpu',copy=True) for n,v in model.named_buffers() if n not in persistent}


def restore_buffers(model,state):
    persistent=set(model.state_dict())
    target={n:b for n,b in model.named_buffers() if n not in persistent}
    if set(target)!=set(state):raise ValueError('Incomplete nonpersistent diagnostic buffers')
    with torch.no_grad():
        for name,tensor in target.items():
            value=state[name]
            if tensor.shape!=value.shape or tensor.dtype!=value.dtype:
                raise ValueError('Nonpersistent buffer mismatch: '+name)
            tensor.copy_(value.to(tensor.device))


def save(cfg,policy,result,last_epoch,source,state,nonpersistent):
    if not enabled(cfg):return None
    if policy not in ('quality','joint') or last_epoch!=60 or not source or not source.get('sha256'):
        raise ValueError('Cannot export incomplete or unverified selected model')
    epoch=int(result['best_epoch'])
    metric=result['selection_metric']
    if not 1<=epoch<=60 or metric not in ('val_quality_SROCC','val_quality_align_mean_SROCC'):
        raise ValueError('Invalid snapshot selection')
    from utils.run_metadata import checkpoint_fingerprint
    path=Path(cfg['output_dir'])/f'semantic_{policy}.pth'
    copied_state=OrderedDict((k,v.detach().to('cpu',copy=True)) for k,v in state.items())
    if hasattr(state,'_metadata'):copied_state._metadata=copy.deepcopy(state._metadata)
    payload=dict(format=FORMAT,training_complete=True,mode='projection_only_v1',
        run_id=cfg['run_id'],cfg=copy.deepcopy(cfg),config_sha256=config_digest(cfg),
        source=source,epoch=epoch,last_epoch=last_epoch,selection_policy=policy,
        selection_metric=metric,best_val=result['best_val'],best_test=result.get('best_test',{}),
        model=copied_state,
        nonpersistent_buffers={k:v.detach().to('cpu',copy=True) for k,v in nonpersistent.items()})
    with path.open('xb') as f:torch.save(payload,f)
    record={k:payload[k] for k in ('format','run_id','config_sha256','epoch','last_epoch','selection_policy','selection_metric')}
    record.update(checkpoint=path.name,sha256=checkpoint_fingerprint(path)['sha256'],source_sha256=source['sha256'])
    with path.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
    return record


def export_comparison(model,cfg,tracker,results,last_epoch,source):
    if not enabled(cfg):return
    if last_epoch!=60:raise ValueError('Cannot export an unfinished comparison')
    for policy,result in results.items():
        record=tracker.best[policy]
        if record['epoch']!=result['best_epoch'] or 'buffers' not in record:
            raise ValueError('Missing selected-epoch diagnostic snapshot')
        save(cfg,policy,result,last_epoch,source,record['state'],record['buffers'])


def load_strict(model,payload):
    if (payload.get('format') not in (FORMAT,'selected_downstream_v1')
            or payload.get('training_complete') is not True or payload.get('last_epoch')!=60
            or payload.get('mode')!='projection_only_v1'):
        raise ValueError('A complete selected downstream snapshot is required; pretraining is not accepted')
    if set(payload.get('model',{}))!=set(model.state_dict()):
        raise ValueError('Missing/foreign full-model tensor keys; no random initialization permitted')
    for name,value in model.state_dict().items():
        source=payload['model'][name]
        if source.shape!=value.shape or source.dtype!=value.dtype:
            raise ValueError('Full-model shape/dtype mismatch: '+name)
    restore_buffers(model,payload.get('nonpersistent_buffers',{}))
    model.load_state_dict(payload['model'],strict=True)
    model.requires_grad_(False);model.eval()
