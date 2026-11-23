"""Six single-factor local-alignment/context controls, twelve downstream runs."""
import copy
import importlib.util
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={
    'alocalaux050':('aux',.15),'alocalaux200':('aux',.60),
    'amatchtau050':('temperature',.5),'amatchtau200':('temperature',2.),
    'actxlr050':('context_lr',.05),'actxlr200':('context_lr',.20),
}


def make_runner():
    spec=importlib.util.spec_from_file_location('_align_local_tuning_protocol',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.__doc__=__doc__;r.ROOT=ROOT;r.VARIANTS=dict(VARIANTS)
    r.QUEUES={
        'low':tuple(VARIANTS)[::2],'high':tuple(VARIANTS)[1::2],
        'aux':tuple(VARIANTS)[:2],'match':tuple(VARIANTS)[2:4],'context':tuple(VARIANTS)[4:],
    }
    r.VERSION='projection_only_v1_v2_align_local_tuning';r.PROTOCOL_DOC='docs/align_local_tuning.md'

    def config(v,ds,split=None):
        if v not in r.VARIANTS or ds not in r.DATASETS:raise ValueError('Unknown local-alignment candidate/dataset')
        if split is not None and split!=42:raise ValueError('Fixed split42')
        c=yaml.safe_load((r.ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text());t=c['train'];m=c['model']
        checkpoint='outputs/'+r.reference()['source']['checkpoint'].split('/outputs/',1)[1]
        if (c['seed']!=42 or c['split_seed']!=42 or t['epochs']!=60 or
                t['pretrained_checkpoint']!=checkpoint or t['pretrained_load_scope']!='representation' or
                t['selection_metric']!='val_quality_SROCC' or t['selection_mode']!='max' or
                m['fusion']['mode']!='projection_only_v1' or m.get('align_branch_control') or
                m.get('align_local_temperature',1.)!=1. or
                t.get('lambda_align_global_aux',.3)!=.3 or t.get('lambda_align_local_aux',.3)!=.3):
            raise ValueError('Original source/mode/selection/pooling required')
        if (t['lambda_align_aux']!=.3 or t['lambda_rank_align']!=.03 or t['lambda_srocc_align']!=.04 or
                t['layerwise_lr']['align_mult']!=1. or t['layerwise_lr']['shared_projection_mult']!=1. or
                t['layerwise_lr'].get('text_context_mult',.1)!=.1 or
                t['lambda_srocc_quality']!={'agiqa3k':.02,'aigciqa2023':.04}[ds]):
            raise ValueError('Restore original baseline optimizer and objective before each candidate')
        n=m['nss_text_gate']
        if n.get('descriptor','legacy')!='legacy' or n['nss_dim']!=6 or n.get('pathway','both')!='both':
            raise ValueError('Original six-dimensional NSS required')
        kind,value=r.VARIANTS[v]
        if kind=='aux':t.update(lambda_align_global_aux=.3,lambda_align_local_aux=value)
        elif kind=='temperature':m['align_local_temperature']=value
        else:t['layerwise_lr']['text_context_mult']=value
        name=f'{ds}_projection_only_v1_v2_{v}_train42_split42'
        c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
        t.update(record_split_manifest=True,record_optimizer_initialization=True)
        return c

    original_verify=r.verify
    def verify(v,ds,split,source):
        original_verify(v,ds,split,source)
        import csv,json,math
        c=config(v,ds,split);_,table=r.paths(v,ds)
        from utils.alignment_controls import alignment_aux_weights
        _,local_weight=alignment_aux_weights(c['train'])
        with table.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            if row['event']!='epoch':continue
            m=json.loads(row['metrics_json'])
            if not math.isclose(m['train_train_weighted_align_patch'],local_weight*m['train_train_align_patch'],rel_tol=1e-3,abs_tol=1e-7):
                raise ValueError('Effective local alignment supervision differs')
            if r.VARIANTS[v][0]=='temperature' and m.get('val_val_align_local_temperature')!=r.VARIANTS[v][1]:
                raise ValueError('Actual local matching temperature differs')
    r.config,r.verify=config,verify
    return r

if __name__=='__main__':make_runner().main()
