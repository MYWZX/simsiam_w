"""Server-only bounded local adaptation; all inherited global/quality state is frozen."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from datasets import build_dataloaders
from models.two_stage_models import load_pretrained_backbone
from train_cm_iqa import build_optimizer,train_one_epoch,evaluate,get_selection_value,unpack_loaders
from utils.experiment_logger import CSVExperimentLogger
from utils.protected_local import (ProtectedLocalIQA,LOCAL_FORMAT,restore_foundation,
    transfer_warm_projection,frozen_digest,write_json_new,sha256)
from utils.run_metadata import collect_run_metadata,optimizer_initialization,checkpoint_fingerprint
from utils.task_gradient_probe import new_region_parameter_record
from utils.train_utils import set_seed


class CleanRichHFLoader:
    """Use only the original human-supervision clean view; never an augmented crop."""
    def __init__(self,loader):self.loader=loader;self.dataset=loader.dataset
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            if 'image_clean' not in batch:raise ValueError('RichHF local warmup requires the clean view')
            yield dict(image=batch['image_clean'],prompt=batch['prompt'],
                quality_score=batch['quality_score'],align_score=batch['align_score'],
                score=torch.zeros_like(batch['align_score']))


def local_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if k.startswith(model.protected_prefixes)}


def fixed_validation(metrics):
    return {k:v for k,v in metrics.items() if k.startswith('val_quality_')
            or k in ('val_align_global_SROCC','val_align_global_PLCC')}


def fit(model,loaders,cfg,upstream,initialization,device):
    """CPU execution is used only by synthetic tests; CLI refuses CPU training."""
    variant=cfg['train']['protected_local_variant'];model.protect(variant)
    before=frozen_digest(model);outdir=Path(cfg['output_dir'])
    optimizer=build_optimizer(cfg,model)
    scaler=torch.amp.GradScaler(device.type,enabled=bool(cfg['train']['amp']) and device.type=='cuda')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['train']['epochs'],eta_min=cfg['scheduler']['min_lr'])
    metadata=collect_run_metadata(upstream,cfg)
    metadata.update(protected_local_version=LOCAL_FORMAT,variant=variant,initialization=initialization,checkpoint_verified_after_load=True,
        frozen_before=before,trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
        optimizer_initialization=optimizer_initialization(optimizer))
    write_json_new(outdir/'initial_metadata.json',metadata)
    log=CSVExperimentLogger(cfg['results_csv'],cfg['run_id'],'protected_local',cfg,str(outdir))
    log.log(event='run_start',epoch=0,metrics={'variant':variant,'upstream':upstream})
    train,val,test=loaders
    reference=evaluate(model,val,device,cfg,split='val')
    if reference.get('val_align_N',0)<2:raise ValueError('Validation alignment labels are required')
    fixed=fixed_validation(reference)
    log.log(event='reference_validation',epoch=0,metrics=reference)
    best=None;best_local=None;best_epoch=0
    for epoch in range(1,cfg['train']['epochs']+1):
        tr=train_one_epoch(model,train,optimizer,scaler,device,cfg,epoch)
        va=evaluate(model,val,device,cfg,split='val');scheduler.step()
        if any(not math.isclose(float(va[k]),float(v),rel_tol=0,abs_tol=1e-6) for k,v in fixed.items()):
            raise RuntimeError('Frozen Quality/global alignment validation outputs changed')
        value,maximize,key=get_selection_value(va,cfg)
        if not maximize:raise ValueError('Local experiment requires maximum validation Align SROCC')
        improved=best is None or value>best
        if improved:best=value;best_epoch=epoch;best_local=local_state(model)
        log.log(event='epoch',epoch=epoch,metrics={**{'train_'+k:v for k,v in tr.items()},
            **{'val_'+k:v for k,v in va.items()},'selection_metric':key,'best_selection_value':best},
            best_epoch=best_epoch,is_best=improved)
    model.load_state_dict(best_local,strict=False)
    after=frozen_digest(model)
    if before!=after:raise RuntimeError('Frozen parameters/buffers changed; reject experiment')
    best_val=evaluate(model,val,device,cfg,split='val')
    best_test=evaluate(model,test,device,cfg,split='test') if test is not None else {}
    final=dict(last_epoch=cfg['train']['epochs'],best_epoch=best_epoch,selection_metric=cfg['train']['selection_metric'],
        best_selection_value=best,best_val=best_val,best_test=best_test,frozen_before=before,frozen_after=after)
    log.log(event='final',epoch=best_epoch,metrics=final,best_epoch=best_epoch)
    checkpoint=dict(format=LOCAL_FORMAT,mode='projection_only_v1',variant=variant,training_complete=True,
        run_id=cfg['run_id'],cfg=cfg,upstream=upstream,initialization=initialization,
        model=best_local,**final)
    path=outdir/'best_local.pth'
    with path.open('xb') as stream:torch.save(checkpoint,stream)
    metadata.update(frozen_after=after,training_complete=True)
    write_json_new(outdir/'run_metadata.json',metadata)
    write_json_new(outdir/'local_checkpoint_manifest.json',dict(format=LOCAL_FORMAT,variant=variant,
        checkpoint=path.name,sha256=sha256(path),selected_epoch=best_epoch,last_epoch=final['last_epoch'],upstream=upstream))
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    if not torch.cuda.is_available():raise RuntimeError('Server CUDA required; cloud CPU training refused')
    from scripts import run_protected_local as protocol
    variant=cfg['train']['protected_local_variant'];ds=cfg['train']['protected_local_dataset']
    if cfg!=protocol.config(variant,ds):raise ValueError('Unexpected local configuration')
    d=Path(cfg['output_dir']);launch=json.loads((d/'launch_manifest.json').read_text())
    if launch['config_sha256']!=sha256(a.config):raise ValueError('Launch config changed')
    if Path(cfg['results_csv']).exists():raise FileExistsError('Preserve existing training CSV')
    upstream=protocol.dependencies(variant,ds)
    if launch['dependencies']!=upstream:raise ValueError('Source changed after launch')
    set_seed(cfg['seed']);device=torch.device('cuda')
    original=protocol.source();source_path=cfg['train']['pretrained_checkpoint']
    build_cfg=copy.deepcopy(cfg)
    build_cfg['model']['global_encoder']['pretrained']=False
    build_cfg['model']['local_encoder']['pretrained']=False
    if variant!='warmup':build_cfg['model']['text_encoder']['clip_pretrained']=None
    model=ProtectedLocalIQA(build_cfg).to(device)
    if variant=='warmup':
        load_pretrained_backbone(model,source_path,load_scope='representation')
        initialized={'new_parameters':new_region_parameter_record(model),'source_scope':'original_representation',
                     'score_heads':'new_RichHF_local_head; discarded_on_downstream_transfer'}
        pretrained_fingerprint=checkpoint_fingerprint(source_path)
        if pretrained_fingerprint['sha256']!=original['sha256']:raise ValueError('Wrong pretraining checkpoint')
    else:
        checkpoint=torch.load(source_path,map_location='cpu',weights_only=False)
        initialized=restore_foundation(model,checkpoint,cfg,expected_run_id=upstream['run_id'],expected_pretrain_sha=original['sha256'])
        initialized['new_parameter_initialization']=new_region_parameter_record(model)
        if variant=='warm':
            warm=protocol.paths('warmup','richhf')[0]/'best_local.pth'
            ck=torch.load(warm,map_location='cpu',weights_only=False)
            initialized['warmup_transfer']=transfer_warm_projection(model,ck,original['sha256'])
            initialized['projection_initialization_after_transfer']=new_region_parameter_record(model)
            for entry in initialized['projection_initialization_after_transfer']:
                entry['initialization']='RichHF_local_warmup'
    loaders=unpack_loaders(build_dataloaders(cfg))
    if variant=='warmup':
        from utils.pretrain_refit import dataset_manifest
        raw=(loaders[0].dataset.datasets[0].dataset,loaders[1].dataset.datasets[0],loaders[2].dataset.datasets[0])
        write_json_new(d/'official_split_manifest.json',dataset_manifest(raw))
        loaders=(CleanRichHFLoader(loaders[0]),CleanRichHFLoader(loaders[1]),None)
    else:
        from utils.split_seed_evidence import record_split
        record_split(cfg,loaders,d)
    with (d/'config_snapshot.yaml').open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
    # Fingerprints checked before and after construction/restoration; no old
    # local head, text context, or Quality state may be silently substituted.
    if protocol.dependencies(variant,ds)!=upstream:raise RuntimeError('Source changed during loading')
    fit(model,loaders,cfg,upstream,initialized,device)
    if protocol.dependencies(variant,ds)!=upstream:raise RuntimeError('Source changed during training')

if __name__=='__main__':main()
