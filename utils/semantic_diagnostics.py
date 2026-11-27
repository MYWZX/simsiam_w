"""Read-only diagnostic provenance, reviewed counterfactuals and frozen forwards."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import torch
from utils.semantic_snapshot import FORMAT,load_strict,config_digest
from utils.run_metadata import checkpoint_fingerprint

ROLES=('baseline','region_tau01')
DATASETS=('agiqa3k','aigciqa2023')
REVIEWS={'candidate','confirmed_original_better','confirmed_equivalent','rejected'}
CATEGORIES={'entity','color','count','spatial','other'}
REPLACEMENTS={
 'entity':{'cat':'dog','dog':'cat','bird':'fish','car':'boat','man':'woman','woman':'man','horse':'cow'},
 'color':{'red':'blue','blue':'red','green':'purple','yellow':'blue','white':'black','black':'white'},
 'count':{'one':'two','two':'three','three':'one','four':'two','1':'2','2':'3','3':'1'},
 'spatial':{'in front of':'behind','on the left':'on the right','left of':'right of','above':'below','below':'above','behind':'in front of'},
}


def read_json(path):return json.loads(Path(path).read_text())
def digest(path):return checkpoint_fingerprint(path)['sha256']
def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def default_specs(root):
    import yaml
    from scripts.run_align_fixed_temperature import make_runner
    r=make_runner();r.ROOT=root;items=[]
    for ds in DATASETS:
        for role in ROLES:
            c=(yaml.safe_load((root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
               if role=='baseline' else r.config('tau010',ds))
            items.append(dict(dataset=ds,role=role,label='historical_'+role,run_id=c['run_id'],run_dir=str(root/c['output_dir'])))
    return items


def expected_config(root,ds,role):
    import yaml
    from scripts.run_align_fixed_temperature import make_runner
    if ds not in DATASETS or role not in ROLES:raise ValueError('Unsupported semantic diagnostic model role')
    return (yaml.safe_load((root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            if role=='baseline' else make_runner().config('tau010',ds))


def validate_config(root,spec,cfg):
    expected=expected_config(root,spec['dataset'],spec['role'])
    if cfg.get('run_id')!=spec['run_id']:raise ValueError('Wrong diagnostic run identity')
    for key in ('model','dataset','seed','split_seed','scheduler'):
        if cfg.get(key)!=expected.get(key):raise ValueError('Diagnostic model configuration differs: '+key)
    harmless={'export_semantic_snapshots','save_selected_downstream','record_run_metadata',
              'record_split_manifest','record_optimizer_initialization'}
    a={k:v for k,v in cfg['train'].items() if k not in harmless}
    b={k:v for k,v in expected['train'].items() if k not in harmless}
    if a!=b:raise ValueError('Diagnostic training protocol differs from the declared role')
    if cfg['dataset'].get('local_multicrop',{}).get('enabled'):raise ValueError('Diagnostic whole-image path only')
    return cfg


def run_evidence(root,spec):
    import yaml
    d=Path(spec['run_dir']);cfg=yaml.safe_load((d/'config_snapshot.yaml').read_text())
    validate_config(root,spec,cfg)
    table=d.parent/(spec['run_id']+'.csv');csv.field_size_limit(50_000_000)
    with table.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    epochs=[x for x in rows if x['event']=='epoch'];finals=[json.loads(x['metrics_json']) for x in rows if x['event']=='final']
    if [int(x['epoch']) for x in epochs]!=list(range(1,61)) or len(finals)!=1 or finals[0].get('last_epoch')!=60:
        raise ValueError('Requires a completed 60-epoch downstream run')
    final=finals[0]
    selected=final['selection_policies']['quality'] if 'selection_policies' in final else final
    if selected['selection_metric']!='val_quality_SROCC':raise ValueError('Only the predeclared Quality-selected model is diagnosed')
    chosen=max(epochs,key=lambda x:json.loads(x['metrics_json'])['val_val_quality_SROCC'])
    if int(chosen['epoch'])!=selected['best_epoch']:raise ValueError('CSV selected epoch differs from validation maximum')
    metadata=read_json(d/'run_metadata.json')
    source=read_json(root/'docs/current_tuning_reference.json')['source']['sha256']
    if not metadata.get('checkpoint_verified_after_load') or metadata['checkpoint']['sha256']!=source:
        raise ValueError('Unverified or foreign pretraining source')
    done=d/'completion_manifest.json'
    if done.is_file():
        completed=read_json(done)
        if completed['csv_sha256']!=digest(table):raise ValueError('Changed run CSV')
        for name,sha in completed['files_sha256'].items():
            if digest(d/name)!=sha:raise ValueError('Changed completed artifact: '+name)
    return cfg,selected,metadata,table


def snapshot_evidence(root,spec):
    cfg,selected,metadata,table=run_evidence(root,spec)
    d=Path(spec['run_dir'])
    options=[(d/'semantic_quality.pth',d/'semantic_quality.json'),
             (d/'best_selected_downstream.pth',d/'selected_checkpoint_manifest.json')]
    pair=next(((p,m) for p,m in options if p.is_file() and m.is_file()),None)
    if pair is None:raise FileNotFoundError('Selected downstream weights/manifest missing; logs or pretraining weights cannot replace them')
    p,m=pair;record=read_json(m)
    if record.get('format') not in (FORMAT,'selected_downstream_v1') or record.get('run_id')!=spec['run_id']:
        raise ValueError('Foreign snapshot manifest')
    epoch=record.get('epoch',record.get('selected_epoch'))
    if epoch!=selected['best_epoch'] or record.get('last_epoch')!=60:raise ValueError('Wrong selected downstream epoch')
    source=record.get('source_sha256',record.get('upstream_sha256'))
    if source!=metadata['checkpoint']['sha256'] or record.get('sha256')!=digest(p):raise ValueError('Snapshot source/SHA256 mismatch')
    return dict(spec=spec,cfg=cfg,selected=selected,metadata=metadata,table=str(table),
                checkpoint=str(p),sha256=record['sha256'],manifest=str(m))


def load_frozen(root,record,device):
    from train_cm_iqa import build_iqa_model
    p=Path(record['checkpoint']);payload=torch.load(p,map_location='cpu',weights_only=True)
    if digest(p)!=record['sha256']:raise ValueError('Snapshot changed while loading')
    if payload.get('run_id')!=record['spec']['run_id'] or payload.get('epoch')!=record['selected']['best_epoch']:
        raise ValueError('Snapshot identity/epoch differs from its completion evidence')
    if payload.get('selection_metric')!='val_quality_SROCC':raise ValueError('Foreign selected policy')
    if payload.get('best_val')!=record['selected']['best_val'] or payload.get('best_test',{})!=record['selected'].get('best_test',{}):
        raise ValueError('Snapshot selected validation/test record differs from completed CSV')
    source=payload.get('source',payload.get('upstream',{}))
    if source.get('sha256')!=record['metadata']['checkpoint']['sha256']:raise ValueError('Foreign snapshot source')
    validate_config(root,record['spec'],payload['cfg'])
    if payload.get('format')==FORMAT and payload.get('config_sha256')!=config_digest(payload['cfg']):
        raise ValueError('Snapshot configuration digest mismatch')
    cfg=copy.deepcopy(payload['cfg']);cfg['train']['pretrained_load_scope']='full'
    model=build_iqa_model(cfg,str(p))
    load_strict(model,payload)
    del payload
    return model.to(device).eval()


def validation_cohort(cfg,run_dir,metadata):
    from datasets import build_dataloaders
    from train_cm_iqa import unpack_loaders
    from utils.refit_protocol import split_manifest,digest as obj_digest,image_identity_order
    c=copy.deepcopy(cfg);c['loader'].update(num_workers=0,pin_memory=False)
    loaders=unpack_loaders(build_dataloaders(c));actual=split_manifest(*loaders)
    path=Path(run_dir)/'data_split_manifest.json'
    if path.is_file():
        recorded=read_json(path)
        if any(recorded[k]!=actual[k] for k in ('indices','dataset_ids_sha256','test_ids_sha256')):
            raise ValueError('Dataset ordering or actual split membership changed')
        mode='recorded_members'
    else:
        recorded=metadata.get('data_split',{})
        if (recorded.get('dataset_ids_sha256')!=actual['dataset_ids_sha256']
                or recorded.get('split_indices_sha256')!=obj_digest(actual['indices'])):
            raise ValueError('Actual validation membership was not recorded/verified for this run; do not infer it from seed alone')
        mode='reconstructed_matches_recorded_digest'
    dataset=loaders[1].dataset.dataset
    return actual,dataset,image_identity_order(dataset),mode


def replacements(prompt):
    """At most one deterministic lexical edit per category, never an automatic label."""
    result=[]
    for category,mapping in REPLACEMENTS.items():
        pattern=r'(?<!\w)('+ '|'.join(re.escape(x) for x in sorted(mapping,key=len,reverse=True))+r')(?!\w)'
        match=re.search(pattern,prompt,re.I)
        if match:
            original=match.group();edited=mapping[original.lower()]
            if original[:1].isupper():edited=edited[:1].upper()+edited[1:]
            result.append(dict(category=category,edited_prompt=prompt[:match.start()]+edited+prompt[match.end():],
                               focus_original=original,focus_edited=edited))
    return result


def validate_cases(document):
    if document.get('version')!='semantic_pairs_v1' or document.get('split')!='val' or document.get('dataset') not in DATASETS:
        raise ValueError('Only versioned validation pairs are supported')
    ids=set()
    for c in document['cases']:
        if not c.get('case_id') or c['case_id'] in ids:raise ValueError('Case IDs must be unique')
        ids.add(c['case_id'])
        if c.get('category') not in CATEGORIES or c.get('review_status') not in REVIEWS:raise ValueError('Unknown case/review category')
        if not c.get('original_prompt') or not c.get('edited_prompt') or c['original_prompt']==c['edited_prompt']:
            raise ValueError('Cases need two distinct nonempty prompts')
        if c['review_status'].startswith('confirmed_') and (not c.get('reviewer','').strip() or not c.get('review_note','').strip()):
            raise ValueError('Confirmed relative ordering/equivalence requires a reviewer and image-grounded note')
    return document


def summarize(rows):
    report={}
    for model in sorted({x['model'] for x in rows}):
        report[model]={}
        for category in ['all',*sorted({x['category'] for x in rows})]:
            selected=[x for x in rows if x['model']==model and (category=='all' or x['category']==category)]
            ordered=[x for x in selected if x['review_status']=='confirmed_original_better']
            equivalent=[x for x in selected if x['review_status']=='confirmed_equivalent']
            item={'confirmed_order_N':len(ordered),'confirmed_equivalent_N':len(equivalent),
                  'candidate_N':sum(x['review_status']=='candidate' for x in selected)}
            for branch in ('global','local','final'):
                ds=[x['delta_'+branch] for x in ordered]
                item[branch]={'mean_drop':sum(ds)/len(ds) if ds else None,
                    'ordering_accuracy':sum(d>1e-6 for d in ds)/len(ds) if ds else None,
                    'tie_rate':sum(abs(d)<=1e-6 for d in ds)/len(ds) if ds else None,
                    'equivalent_mean_abs_change':sum(abs(x['delta_'+branch]) for x in equivalent)/len(equivalent) if equivalent else None}
            report[model][category]=item
    return report


def infer(model,image,prompt,focus=''):
    from models.alignment_evidence import truncation_flags
    with torch.inference_mode():
        if bool(truncation_flags(model.backbone.text_encoder,[prompt],image.device)[0]):
            raise ValueError('Diagnostic Prompt would be truncated; shorten and re-review this case')
    captured={};handle=None
    if model.align_region_word is not None:
        handle=model.align_region_word.register_forward_hook(lambda m,a,o:captured.update(weights=o[2].detach().float().cpu()[0]))
    try:
        with torch.inference_mode():out=model.forward_single(image,[prompt])
    finally:
        if handle is not None:handle.remove()
    result={name:float(out[key].reshape(-1)[0]) for name,key in dict(quality='s_quality',global_score='s_align_global',
                local_score='s_align_patch',final_score='s_align',gate_global='align_gate').items()}
    if not all(math.isfinite(x) for x in result.values()):raise ValueError('Nonfinite diagnostic scores')
    if captured:
        weights=captured['weights'];ids=out['text_token_ids'][0].detach().cpu()
        active=weights.sum(0)>0;positions=active.nonzero().flatten().tolist()
        matrix=weights[:,active];tokenizer=model.backbone.text_encoder.tokenizer
        tokens=[tokenizer.decode([int(ids[i])]) for i in positions]
        focused=[]
        if focus:
            word=tokenizer([focus])[0];pattern=word[1:int(word.argmax())].tolist()
            full=ids.tolist()
            if pattern:
                for i in range(len(full)-len(pattern)+1):
                    if full[i:i+len(pattern)]==pattern:focused.extend(j for j in range(i,i+len(pattern)) if j in positions)
        attention=weights[:,sorted(set(focused))].mean(1) if focused else matrix.mean(1)
        scope='focus_BPE_mean' if focused else 'prompt_mean_focus_unavailable'
    else:
        attention=out['align_patch_weights'][0].detach().float().cpu();matrix=attention[:,None];tokens=[]
        scope='whole_prompt_only_no_token_map'
    side=math.isqrt(len(attention))
    if side*side!=len(attention):raise ValueError('Unsupported non-square diagnostic token grid')
    result.update(map=attention.reshape(side,side).numpy(),token_maps=matrix.numpy(),tokens=tokens,map_scope=scope)
    return result
