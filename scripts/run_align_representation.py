"""Baseline diagnostics/selection vs CLIP region-word alignment; four downstream runs."""
import importlib.util
import json
import math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={'aprobe':('diagnostic',1), 'aregionword':('region_word',1)}


def verify_comparison(c, rows, groups):
    epochs=[r for r in rows if r['event']=='epoch']
    if [int(r['epoch']) for r in epochs]!=list(range(1,61)) or any(r['event']=='resume' for r in rows):
        raise ValueError('Expected one complete fresh 60-epoch trajectory')
    finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
    if len(finals)!=1 or finals[0].get('last_epoch')!=60 or finals[0].get('comparison_complete') is not True:
        raise ValueError('Missing completed selection comparison')
    records=[json.loads(r['metrics_json']) for r in rows if r['event']=='optimizer_initialization']
    if records!=[{'groups':groups}]:raise ValueError('Missing optimizer initialization event')
    if set(finals[0].get('selection_policies',{}))!={'quality','joint'}:
        raise ValueError('Expected exactly two final selection policies')
    for name,key in c['train']['selection_policies'].items():
        values=[]
        for row in epochs:
            m=json.loads(row['metrics_json'])
            if m.get('val_val_align_N',0)<2:raise ValueError('Validation Align labels required')
            q,a=m['val_val_quality_SROCC'],m['val_val_align_SROCC']
            if not all(math.isfinite(x) and -1<=x<=1 for x in (q,a)):raise ValueError('Invalid validation correlation')
            values.append(q if name=='quality' else (q+a)/2)
        index=max(range(len(values)),key=values.__getitem__)
        result=finals[0]['selection_policies'][name]
        selected=[r for r in rows if r['event']==name+'_selected']
        if (result['best_epoch']!=index+1 or result['selection_metric']!=key
                or not math.isclose(result['best_selection_value'],values[index],abs_tol=1e-12)
                or len(selected)!=1 or int(selected[0]['epoch'])!=index+1
                or json.loads(selected[0]['metrics_json'])!=result):
            raise ValueError('Selection policy/selected-test identity mismatch')
        chosen=json.loads(epochs[index]['metrics_json'])
        for task in ('quality','align'):
            if result['best_val'].get('val_'+task+'_SROCC')!=chosen['val_val_'+task+'_SROCC']:
                raise ValueError('Selected validation metrics do not match the chosen epoch')
    return epochs


def make_runner():
    spec=importlib.util.spec_from_file_location('_align_representation_protocol',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.__doc__=__doc__;r.ROOT=ROOT;r.VARIANTS=dict(VARIANTS)
    r.QUEUES={'probe':('aprobe',),'region':('aregionword',)}
    r.VERSION='projection_only_v1_v2_align_representation'
    r.PROTOCOL_DOC='docs/align_representation.md'
    r.CONFIG_SUMMARY='train42/split42; original source/NSS; Quality + joint-mean selection; bounded same-forward probes'
    r.FILES=(*r.FILES,'task_gradient_probe.jsonl')
    def config(v,ds,split=None):
        if v not in VARIANTS or ds not in r.DATASETS:raise ValueError('Only the two named cases on 3K/2023')
        if split not in (None,42):raise ValueError('Fixed split42')
        c=yaml.safe_load((r.ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())
        t=c['train'];m=c['model']
        expected='outputs/'+r.reference()['source']['checkpoint'].split('/outputs/',1)[1]
        if (c['seed']!=42 or c['split_seed']!=42 or t['epochs']!=60
                or t['pretrained_checkpoint']!=expected or t['pretrained_load_scope']!='representation'
                or t['selection_metric']!='val_quality_SROCC' or t['selection_mode']!='max'
                or m['fusion']['mode']!='projection_only_v1' or m.get('align_branch_control')
                or m.get('align_region_word') or m.get('align_local_temperature',1)!=1
                or m['nss_text_gate']['nss_dim']!=6 or m['nss_text_gate'].get('descriptor','legacy')!='legacy'):
            raise ValueError('Original v2 source/configuration required')
        name=f'{ds}_projection_only_v1_v2_{v}_train42_split42'
        c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
        t.update(record_split_manifest=True,record_optimizer_initialization=True,
                 compare_selection_policies=True,selection_comparison_version='quality_mean_v1',
                 selection_policies={'quality':'val_quality_SROCC','joint':'val_quality_align_mean_SROCC'},
                 early_stopping_enabled=False,
                 task_gradient_probe={'version':'same_forward_v1','epochs':[1,10,30,60],'batches':2})
        if v=='aregionword':m['align_region_word']={'version':'clip_region_word_v1'}
        return c
    r.config=config;r.verify_selection_records=verify_comparison
    original=r.verify
    def verify(v,ds,split,source):
        original(v,ds,split,source)
        d,_=r.paths(v,ds)
        records=[json.loads(line) for line in (d/'task_gradient_probe.jsonl').read_text().splitlines()]
        if [(x['epoch'],x['batch']) for x in records]!=[(e,b) for e in (1,10,30,60) for b in (1,2)]:
            raise ValueError('Missing or repeated gradient diagnostics')
        for row in records:
            if row['version']!='same_forward_v1' or row['model_mode']!='train' or set(row['groups'])!={'clip_visual','resnet_local','global_projection'}:
                raise ValueError('Wrong gradient probe semantics')
            for group in row['groups'].values():
                if group.get('status')!='ok':raise ValueError('Unavailable/nonfinite task gradient diagnostics')
                for key in ('quality_norm','align_norm'):
                    value=group.get(key)
                    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                        raise ValueError('Invalid gradient norm')
                cosine=group.get('cosine')
                if cosine is not None and (not math.isfinite(cosine) or not -1<=cosine<=1):
                    raise ValueError('Invalid gradient cosine')
            if v=='aregionword' and row['groups']['resnet_local']['align_norm']!=0:
                raise ValueError('Region-word alignment unexpectedly depends on ResNet')
        m=r.base.read_json(d/'run_metadata.json')
        params=m.get('region_word_initialization',[])
        expected_shapes={'align_region_word.image_projection.weight':[256,1024],
                         'align_region_word.text_projection.weight':[256,768]}
        if v=='aregionword':
            if m.get('region_word_version')!='clip_region_word_v1' or {p['name']:p['shape'] for p in params}!=expected_shapes:
                raise ValueError('Missing explicit new parameter initialization')
            if any(p['initialization']!='new_random' or len(p['sha256'])!=64 for p in params):
                raise ValueError('Missing initialized parameter fingerprint')
        elif params:raise ValueError('Baseline unexpectedly added parameters')
    r.verify=verify
    return r

if __name__=='__main__':make_runner().main()
