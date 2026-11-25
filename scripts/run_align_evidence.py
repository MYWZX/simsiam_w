"""Sequential evidence ablations, with independent original-NSS and kurtosis arms."""
import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_align_representation as representation
from scripts import run_nss_kurtosis as kurtosis
VARIANTS={'diag':0,'temperature':1,'tokenmean':2,'coverage':3,'reference':4}


def make_runner(nss='legacy'):
    if nss not in ('legacy','kurt6'):raise ValueError('Unknown NSS arm')
    spec=importlib.util.spec_from_file_location('_evidence_'+nss,ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.ROOT=ROOT;r.VARIANTS=dict(VARIANTS);r.QUEUES={'ordered':tuple(VARIANTS)}
    r.VERSION=f'projection_only_v1_v2_align_evidence_v1_{nss}'
    r.PROTOCOL_DOC='docs/align_evidence.md'
    r.CONFIG_SUMMARY='60 epochs, Quality/joint selection; diagnostics then bounded temperature, token mean, coverage, native CLIP reference'
    r.verify_selection_records=representation.verify_comparison
    def config(v,ds,split=None):
        if v not in VARIANTS or ds not in r.DATASETS or split not in (None,42):raise ValueError('Only named arms on 3K/2023 split42')
        c=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())
        if nss=='kurt6':
            c['model']['nss_text_gate'].update(descriptor='legacy_kurt6_v1',nss_dim=6,pathway='both')
            c['train']['pretrained_checkpoint']=kurtosis.expected_config('kurt6')['output_dir']+'/best_component_anchor_pretrain.pth'
        name=f'{ds}_projection_only_v1_v2_evidence_{nss}_{v}_train42_split42'
        c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
        c['train'].update(record_split_manifest=True,record_optimizer_initialization=True,
              compare_selection_policies=True,selection_comparison_version='quality_mean_v1',
              selection_policies={'quality':'val_quality_SROCC','joint':'val_quality_align_mean_SROCC'},
              early_stopping_enabled=False)
        level=VARIANTS[v]
        e={'version':'align_evidence_v1','diagnostics':True,
           'temperature':{'mode':'fixed' if level==0 else 'learned','initial':1.,'minimum':.03,'maximum':2.}}
        if level>=2:
            c['model']['align_region_word']={'version':'clip_region_word_v1'}
            e['coverage']={'mode':'softmin' if level>=3 else 'mean','temperature':.1,'lexical_mask':True}
        if level>=4:e['native_reference']={'enabled':True,'initial_mix':.05}
        c['model']['align_evidence']=e
        return c
    r.config=config
    original_source=r.source;original_control=r.control;original_reference=r.reference
    if nss=='kurt6':
        r.source=lambda:kurtosis.make_runner().completed('pretrain')
        def control(ds,split,fp):
            control_runner=kurtosis.make_runner()
            if control_runner.completed(ds)!=fp:raise ValueError('Kurtosis control source mismatch')
            return control_runner.paths(ds)
        r.control=control
        def reference():
            data=copy.deepcopy(original_reference())
            c=kurtosis.expected_config('kurt6')
            data['source']={'version':'projection_only_v1_ctx5_legacy_kurt6_v1',
                            'checkpoint':c['output_dir']+'/best_component_anchor_pretrain.pth',
                            'status':'resolved and SHA256 verified at runtime; fresh pretraining required'}
            return data
        r.reference=reference
    original_verify=r.verify
    def verify(v,ds,split,fp):
        original_verify(v,ds,split,fp)
        d,table=r.paths(v,ds);meta=r.base.read_json(d/'run_metadata.json')
        record=meta.get('alignment_evidence_initialization',{})
        if record.get('config')!=config(v,ds)['model']['align_evidence']:
            raise ValueError('Evidence configuration/init metadata mismatch')
        expected=set()
        if VARIANTS[v]>=1:expected.add('align_evidence.temperature_logit')
        if VARIANTS[v]>=4:expected.update('align_evidence.'+x for x in ('clip_scale','clip_bias','clip_mix_logit'))
        params=record.get('parameters',[])
        if {p['name'] for p in params}!=expected or any(p['initialization']!='new_explicit' or len(p['sha256'])!=64 for p in params):
            raise ValueError('Unexpected/missing explicit evidence parameter initialization')
        import csv
        with table.open(newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            if row['event']!='epoch':continue
            m=json.loads(row['metrics_json'])
            for key in ('spatial_entropy','spatial_max','temperature','prompt_truncated','s_align_MAE'):
                value=m.get('val_val_evidence_'+key)
                if value is None or not math.isfinite(value):raise ValueError('Missing/nonfinite evidence diagnostics: '+key)
            if not .03<m['val_val_evidence_temperature']<2.:raise ValueError('Temperature out of bounds')
        if VARIANTS[v]>=2:
            params=meta.get('region_word_initialization',[])
            if {p['name']:p['shape'] for p in params}!={'align_region_word.image_projection.weight':[256,1024],
                                                      'align_region_word.text_projection.weight':[256,768]}:
                raise ValueError('Region-token new parameter provenance missing')
    r.verify=verify
    r.pack_extra_records=lambda fp:{'evidence_protocol.json':{'source':fp,'nss':nss,
       'ordered_variants':VARIANTS,'native_reference':'current paired CLIP; same five contexts; no frozen teacher',
       'comparisons':['legacy/diag vs kurt6/diag isolates NSS','diag -> temperature',
                      'temperature -> tokenmean changes local path','tokenmean -> coverage','coverage -> reference']}}
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('nss',choices=('legacy','kurt6'))
    p.add_argument('action',choices=tuple(VARIANTS)+('all','pack','check-configs'))
    p.add_argument('--dataset',choices=('agiqa3k','aigciqa2023'));p.add_argument('--dry-run',action='store_true')
    p.add_argument('--continue',dest='completed',action='store_true');a=p.parse_args();os.chdir(ROOT)
    r=make_runner(a.nss)
    if a.action=='all':r.run_all('all',a.dry_run,a.completed,a.dataset)
    elif a.action=='check-configs':
        for ds in r.DATASETS:
            for v in VARIANTS:r.config(v,ds)
        print('10 fixed evidence configs valid:',a.nss)
    elif a.action=='pack':
        if a.dry_run or a.dataset:p.error('Packing covers the complete arm')
        r.pack()
    else:
        if not a.dataset:p.error('A single variant requires --dataset')
        if a.dry_run:print(json.dumps(r.config(a.action,a.dataset),indent=2));return
        fp=r.source();r.control(a.dataset,None,fp)
        if a.completed and any(x.exists() for x in r.paths(a.action,a.dataset)):
            r.verify(a.action,a.dataset,None,fp)
        else:r.run(a.action,a.dataset)
if __name__=='__main__':main()
