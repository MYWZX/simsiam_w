"""Validation-derived, immutable train+val refit with a single final test."""
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from utils.data_protocols import loader_rng_kwargs, fit_1k_train_scaler
from utils.run_metadata import checkpoint_fingerprint


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical_cfg(cfg):
    c=copy.deepcopy(cfg)
    for key in ['run_id','output_dir','results_csv','source_config']:c.pop(key,None)
    c.get('train',{}).pop('refit_plan',None)
    for key in ['fitted_mos_normalization','refit_split_manifest']:c.get('dataset',{}).pop(key,None)
    return c


class ImageSplitOverlapError(ValueError):
    def __init__(self, report):
        self.report = report
        counts = {key: value['image_count'] for key, value in report['cross_split'].items()
                  if value['image_count']}
        super().__init__(f'Repeated image IDs across splits: {counts}; inspect the overlap report')


def image_identity_order(dataset):
    """Resolve active loaders' actual image paths without opening image pixels."""
    if hasattr(dataset, 'df'):
        names = dataset.df['name'].astype(str).tolist()
        resolver = getattr(dataset, '_resolve_image_path', None) or getattr(dataset, '_path', None)
        return [str(Path(resolver(name)).resolve()) for name in names] if resolver else names
    if hasattr(dataset, 'images'):
        return [str(Path(path).resolve()) for path in dataset.images]
    raise ValueError('Unsupported dataset identity schema')


def split_overlap_report(loaders, sample_limit=20):
    names = ('train', 'val', 'test')
    orders = [image_identity_order(loader.dataset.dataset) for loader in loaders]
    if orders[0] != orders[1] or orders[0] != orders[2]:
        raise ValueError('Train/eval actual image ordering mismatch')
    members = {}
    for name, loader in zip(names, loaders):
        members[name] = {}
        for i in loader.dataset.indices:
            i = int(i)
            members[name].setdefault(orders[0][i], []).append(i)
    def rows(split, indices):
        dataset = loaders[names.index(split)].dataset.dataset
        result = []
        for i in indices[:sample_limit]:
            detail = {'row_index': i}
            if hasattr(dataset, 'df'):
                row = dataset.df.iloc[i]
                for key in ('name', 'MOS', 'mos_quality', 'mos_align', 'prompt'):
                    if key in row:
                        detail[key] = str(row[key])[:200]
            result.append(detail)
        return result
    report = {'split_sizes': {name: len(loader.dataset) for name,loader in zip(names,loaders)},
              'cross_split': {}, 'within_split_duplicate_image_counts': {
                  name: sum(len(v)>1 for v in group.values()) for name,group in members.items()},
              'identity': 'resolved image paths for active datasets; row IDs for synthetic datasets',
              'sample_limit': sample_limit}
    for left, right in [('train','val'), ('train','test'), ('val','test')]:
        overlap = sorted(set(members[left]) & set(members[right]))
        report['cross_split'][left+'_'+right] = {'image_count': len(overlap), 'examples': [
            {'image_path': path, left: rows(left,members[left][path]),
             right: rows(right,members[right][path])} for path in overlap[:sample_limit]]}
    return report


def split_manifest(train_loader,val_loader,test_loader):
    if test_loader is None:raise ValueError('Refit needs an original held-out test split')
    loaders=[train_loader,val_loader,test_loader]
    if not all(isinstance(x.dataset,Subset) for x in loaders):raise ValueError('Refit supports explicit downstream Subset splits only')
    def ids(ds):
        if hasattr(ds,'df'):return ds.df['name'].astype(str).tolist()
        if hasattr(ds,'images'):return [str(x.relative_to(ds.image_root)) for x in ds.images]
        raise ValueError('Unsupported dataset identity schema')
    ordered=[ids(x.dataset.dataset) for x in loaders]
    if ordered[0]!=ordered[1] or ordered[0]!=ordered[2]:raise ValueError('Train/eval dataset ordering mismatch')
    indices={name:[int(i) for i in loader.dataset.indices] for name,loader in zip(['train','val','test'],loaders)}
    sets={k:set(v) for k,v in indices.items()}
    if any(len(sets[k])!=len(v) for k,v in indices.items()):raise ValueError('Repeated split indices')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if sets[a]&sets[b]:raise ValueError('Split overlap')
    if sets['train']|sets['val']|sets['test']!=set(range(len(ordered[0]))):raise ValueError('Split does not cover the dataset exactly')
    report = split_overlap_report(loaders)
    if any(pair['image_count'] for pair in report['cross_split'].values()):
        raise ImageSplitOverlapError(report)
    return dict(dataset_ids_sha256=digest(ordered[0]),indices=indices,
        test_ids_sha256=digest([ordered[0][i] for i in indices['test']]),
        train_batches=len(train_loader),split_sizes={k:len(v) for k,v in indices.items()})


def validation_budget(csv_path,cfg,selector):
    """Read epoch validation values only; ignore test/final-selected metrics."""
    with open(csv_path,encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if sum(r['event']=='run_start' for r in rows)!=1 or sum(r['event']=='final' for r in rows)!=1:
        raise ValueError('Reference CSV must contain one completed run')
    best=None;records=[]
    for row in rows:
        if row['event']!='epoch':continue
        metrics=json.loads(row['metrics_json'])
        validation={k[4:]:v for k,v in metrics.items() if k.startswith('val_')}
        value,maximize,key=selector(validation,cfg)
        steps=metrics.get('train_processed_steps')
        if isinstance(steps,bool) or not isinstance(steps,int) or steps<1:raise ValueError('Reference run lacks recorded training-step budget')
        epoch=int(row['epoch']);records.append((epoch,steps))
        if best is None or (value>best['value'] if maximize else value<best['value']):
            best=dict(epoch=epoch,value=value,metric=key)
    if not records or [e for e,_ in records]!=list(range(1,len(records)+1)):raise ValueError('Incomplete or repeated reference epochs')
    return dict(**best,total_steps=sum(s for e,s in records if e<=best['epoch']),epoch_steps=[s for _,s in records])


def make_plan(cfg,csv_path,loaders,selector):
    if cfg.get('train',{}).get('protocol_version')!='corrected_v2' or not cfg['model'].get('clip_quick_gelu'):
        raise ValueError('Refit requires corrected_v2 QuickGELU development results')
    if cfg['train'].get('progressive_unfreeze',{}).get('enabled',False):raise ValueError('Progressive refit is not supported')
    if cfg['train'].get('compare_selection_policies',False):raise ValueError('Fix one validation policy before refit')
    manifest=split_manifest(*loaders);budget=validation_budget(csv_path,cfg,selector)
    if any(s!=manifest['train_batches'] for s in budget['epoch_steps']):raise ValueError('Reference batches do not match reconstructed training split')
    plan=dict(version=1,protocol='train_val_fixed_budget',config_sha256=digest(canonical_cfg(cfg)),
        reference_csv=str(Path(csv_path).resolve()),reference_csv_sha256=file_digest(csv_path),
        pretrained=checkpoint_fingerprint(cfg['train']['pretrained_checkpoint']),
        source_epochs=int(cfg['train']['epochs']),validation_selection=budget,
        total_steps=budget['total_steps'],step_unit='processed_training_batches',
        split=manifest)
    plan['plan_sha256']=digest(plan)
    return plan


def load_plan(path,cfg):
    plan=json.loads(Path(path).read_text())
    expected=plan.get('plan_sha256');payload=dict(plan);payload.pop('plan_sha256',None)
    if plan.get('version')!=1 or expected!=digest(payload):raise ValueError('Refit plan integrity check failed')
    if digest(canonical_cfg(cfg))!=plan['config_sha256']:raise ValueError('Refit scientific config differs from development config')
    if file_digest(plan['reference_csv'])!=plan['reference_csv_sha256']:raise ValueError('Reference CSV changed after planning')
    if checkpoint_fingerprint(cfg['train']['pretrained_checkpoint'])!=plan['pretrained']:raise ValueError('Pretrained source changed after planning')
    return plan


def merge_train_validation(train_loader,val_loader,test_loader,cfg,plan):
    current=split_manifest(train_loader,val_loader,test_loader)
    if current!=plan['split']:raise ValueError('Dataset identities or original split changed after planning')
    indices=current['indices']['train']+current['indices']['val']
    base=train_loader.dataset.dataset
    if cfg['dataset']['type']=='agiqa1k_iqa':
        fit_1k_train_scaler(base,val_loader.dataset.dataset,indices,cfg,'train_plus_val')
    merged=Subset(base,indices) # Train transforms for every newly included sample.
    options=cfg['loader']
    result=DataLoader(merged,batch_size=options.get('batch_size',8),shuffle=True,
        num_workers=options.get('num_workers',4),pin_memory=options.get('pin_memory',True),
        drop_last=True,**loader_rng_kwargs(cfg,'train'))
    if not len(result):raise ValueError('Merged training loader has no full batch')
    cfg['dataset']['refit_split_manifest']=dict(original_split_sizes=current['split_sizes'],
        actual_train_count=len(indices),actual_val_count=0,test_count=len(test_loader.dataset),
        test_ids_sha256=current['test_ids_sha256'],plan_sha256=plan['plan_sha256'])
    return result,None


class LimitedLoader:
    def __init__(self,loader,limit):self.loader=loader;self.limit=min(len(loader),limit);self.dataset=loader.dataset
    def __len__(self):return self.limit
    def __iter__(self):return itertools.islice(self.loader,self.limit)


def step_lrs(initial,minimum,step,original_batches,planned_epochs):
    # Preserve the development schedule in original-epoch units, including
    # its stopping point, rather than forcing cosine to finish at refit end.
    epoch=min(planned_epochs,step//original_batches)
    factor=.5*(1.+math.cos(math.pi*epoch/planned_epochs))
    return [minimum+(lr-minimum)*factor for lr in initial]


def run_fixed_budget_refit(*,model,train_loader,test_loader,optimizer,device,cfg,plan,logger,train_epoch,evaluate_fn,metadata_path):
    if test_loader is None:raise ValueError('No test split for final refit report')
    scaler=GradScaler(device.type,enabled=bool(cfg['train'].get('amp',True)) and device.type=='cuda')
    initial=[float(g['lr']) for g in optimizer.param_groups]
    steps,updates,epoch=0,0,0
    def after_step(successful):
        nonlocal steps,updates
        steps+=1;updates+=int(successful)
        lrs=step_lrs(initial,float(cfg.get('scheduler',{}).get('min_lr',1e-6)),steps,
            plan['split']['train_batches'],plan['source_epochs'])
        for group,lr in zip(optimizer.param_groups,lrs):group['lr']=lr
    logger.log(event='refit_start',epoch=0,metrics={'plan_sha256':plan['plan_sha256'],
        'planned_steps':plan['total_steps'],'split':cfg['dataset']['refit_split_manifest']})
    while steps<plan['total_steps']:
        epoch+=1;before=steps
        tr=train_epoch(model,LimitedLoader(train_loader,plan['total_steps']-steps),optimizer,scaler,
            device,cfg,epoch,optimizer_step_callback=after_step)
        if steps==before:raise RuntimeError('Refit made no training progress')
        logger.log(event='refit_epoch',epoch=epoch,metrics={'train':tr,'training_steps':steps,'optimizer_updates':updates})
    if steps!=plan['total_steps']:raise RuntimeError('Refit exceeded step budget')
    # The only test evaluation in this execution mode. No validation selection.
    test=evaluate_fn(model,test_loader,device,cfg,split='test')
    logger.log(event='final',epoch=epoch,metrics={'protocol':'train_val_fixed_budget',
        'final_epoch':epoch,'training_steps':steps,'optimizer_updates':updates,'amp_skipped_steps':steps-updates,
        'plan_sha256':plan['plan_sha256'],'test':test,'run_metadata_path':metadata_path,
        'best_checkpoint':'','last_checkpoint':''})
