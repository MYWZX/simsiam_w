"""Four downstream controls: identical region/BPE-token route, fixed tau=1 vs .1."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_align_evidence as evidence
from scripts.run_align_40plus20 import verify_lrs,verify_step_budget
VARIANTS={'tau100':1.,'tau010':.1}
VERSION='projection_only_v1_v2_align_fixed_temperature_v1'


def read_rows(table):
    csv.field_size_limit(50_000_000)
    with table.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def verify_diagnostics(c,rows):
    expected=c['model']['align_evidence']['temperature']['initial']
    required=('temperature','spatial_entropy','spatial_max','prompt_truncated',
              's_align_MAE','s_align_global_MAE','s_align_patch_MAE','token_match_mean','token_match_min')
    def check(metrics,prefix):
        for key in required:
            value=metrics.get(prefix+'evidence_'+key)
            if value is None or not math.isfinite(value):
                raise ValueError('Missing/nonfinite fixed-temperature diagnostic: '+key)
        if not math.isclose(metrics[prefix+'evidence_temperature'],expected,rel_tol=1e-6,abs_tol=1e-7):
            raise ValueError('Actual fixed matching temperature differs from configuration')
        for key in ('spatial_entropy','spatial_max','prompt_truncated'):
            if not -1e-6<=metrics[prefix+'evidence_'+key]<=1+1e-6:
                raise ValueError('Diagnostic fraction out of range: '+key)
    for row in rows:
        if row['event']=='epoch':
            metrics=json.loads(row['metrics_json']);check(metrics,'val_val_')
            if not math.isfinite(metrics.get('val_val_quality_loss',float('nan'))):
                raise ValueError('Missing/nonfinite validation quality MSE')
        if row['event']=='final':
            for result in json.loads(row['metrics_json'])['selection_policies'].values():
                check(result['best_val'],'val_');check(result['best_test'],'test_')


def verify_initialization(c,meta):
    init=meta.get('alignment_evidence_initialization',{})
    if (init.get('version')!='align_evidence_v1' or init.get('config')!=c['model']['align_evidence']
            or init.get('parameters')!=[]):
        raise ValueError('Fixed temperature must add no learned evidence/calibration parameters')
    expected={'align_region_word.image_projection.weight':[256,1024],
              'align_region_word.text_projection.weight':[256,768]}
    params=meta.get('region_word_initialization',[])
    if (meta.get('region_word_version')!='clip_region_word_v1'
            or {p['name']:p['shape'] for p in params}!=expected
            or any(p['initialization']!='new_random' or len(p['sha256'])!=64 for p in params)):
        raise ValueError('Missing explicit region/token projection initialization fingerprints')


def make_runner():
    spec=importlib.util.spec_from_file_location('_fixed_temperature_protocol',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.ROOT=ROOT;r.VARIANTS=dict(VARIANTS);r.QUEUES={'unit':('tau100',),'sharp':('tau010',)}
    r.VERSION=VERSION;r.PROTOCOL_DOC='docs/align_fixed_temperature.md'
    r.CONFIG_SUMMARY='original six-channel NSS/source; fixed tau=1 vs .1; same region-token mean route; 60 epochs; Q/joint selection'
    r.verify_selection_records=evidence.representation.verify_comparison
    def config(v,ds,split=None):
        if v not in VARIANTS or ds not in r.DATASETS or split not in (None,42):
            raise ValueError('Only tau100/tau010 on 3K/2023 split42')
        c=copy.deepcopy(evidence.make_runner('legacy').config('tokenmean',ds,split))
        t,m=c['train'],c['model']
        expected_source='outputs/'+r.reference()['source']['checkpoint'].split('/outputs/',1)[1]
        if (c['seed']!=42 or c['split_seed']!=42 or t['epochs']!=60
                or t['pretrained_checkpoint']!=expected_source or t['pretrained_load_scope']!='representation'
                or m['fusion']['mode']!='projection_only_v1'
                or m['nss_text_gate'].get('descriptor','legacy')!='legacy'
                or m['nss_text_gate']['nss_dim']!=6 or m['nss_text_gate'].get('pathway','both')!='both'
                or t['layerwise_lr']['shared_projection_mult']!=1.
                or t['layerwise_lr']['fusion_mult']!=.1 or t['lr']!=1e-4
                or t['layerwise_lr']['align_mult']!=1.
                or t['lambda_srocc_quality']!={'agiqa3k':.02,'aigciqa2023':.04}[ds]):
            raise ValueError('Restore the original v2 NSS/source/optimizer protocol')
        m['align_evidence']['temperature'].update(mode='fixed',initial=VARIANTS[v])
        name=f'{ds}_{VERSION}_{v}_train42_split42'
        c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
        return c
    r.config=config
    original=r.verify
    def verify(v,ds,split,source):
        d,table=r.paths(v,ds,split)
        if not (d/'completion_manifest.json').is_file():
            raise RuntimeError('Incomplete or pending fixed-temperature run; preserve output and inspect logs: '+str(d))
        original(v,ds,split,source)
        c=config(v,ds,split);rows=read_rows(table);meta=r.base.read_json(d/'run_metadata.json')
        verify_initialization(c,meta);verify_diagnostics(c,rows)
        verify_lrs(rows,meta['optimizer_initialization'])
        split_record=r.base.read_json(d/'data_split_manifest.json');verify_step_budget(rows,split_record)
        other=next(x for x in VARIANTS if x!=v);od,ot=r.paths(other,ds,split)
        if (od/'completion_manifest.json').is_file():
            # Original verifier does not recurse into this pair check.
            original(other,ds,split,source)
            om=r.base.read_json(od/'run_metadata.json')
            verify_initialization(config(other,ds,split),om)
            if meta['region_word_initialization']!=om['region_word_initialization']:
                raise ValueError('Paired initial region/token projection fingerprints differ')
            if meta['optimizer_initialization']!=om['optimizer_initialization']:
                raise ValueError('Paired optimizer groups/LRs/parameter counts differ')
            osplit=r.base.read_json(od/'data_split_manifest.json')
            if any(split_record[k]!=osplit[k] for k in ('indices','dataset_ids_sha256','test_ids_sha256')):
                raise ValueError('Paired actual split memberships differ')
    r.verify=verify
    r.pack_extra_records=lambda fp:{'fixed_temperature_protocol.json':{
        'version':VERSION,'source':fp,'fixed_temperatures':VARIANTS,
        'nss':'legacy six-dimensional unchanged','region_token_pool':'mean with lexical BPE mask',
        'new_trainable_parameters':'two explicitly initialized region/token projections; no learned temperature',
        'selection':'quality and arithmetic-mean joint on one 60-epoch trajectory',
        'historical_learned_temperature_results_reused':False}}
    return r

if __name__=='__main__':make_runner().main()
