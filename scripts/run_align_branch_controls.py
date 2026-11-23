"""Three downstream-only alignment branch controls; six original-source runs."""
import copy
import importlib.util
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={
    'aglobal':{'version':'alignment_branch_v1','output':'global','local_pool':'text'},
    'ahalf':{'version':'alignment_branch_v1','output':'half','local_pool':'text'},
    'auniform':{'version':'alignment_branch_v1','output':'learned','local_pool':'uniform'},
}


def make_runner():
    spec=importlib.util.spec_from_file_location('_align_branch_protocol',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.__doc__=__doc__;r.ROOT=ROOT;r.VARIANTS=copy.deepcopy(VARIANTS)
    r.QUEUES={'gate':('aglobal','ahalf'),'pool':('auniform',)}
    r.VERSION='projection_only_v1_v2_align_branch';r.PROTOCOL_DOC='docs/align_branch_controls.md'

    def config(v,ds,split=None):
        if v not in r.VARIANTS or ds not in r.DATASETS:raise ValueError('Unknown branch condition/dataset')
        if split is not None and split!=42:raise ValueError('Fixed split seed42')
        c=yaml.safe_load((r.ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text());t=c['train'];m=c['model']
        checkpoint='outputs/'+r.reference()['source']['checkpoint'].split('/outputs/',1)[1]
        if (c['seed']!=42 or c['split_seed']!=42 or t['epochs']!=60 or
                t['pretrained_checkpoint']!=checkpoint or t['pretrained_load_scope']!='representation' or
                t['selection_metric']!='val_quality_SROCC' or t['selection_mode']!='max' or
                m['fusion']['mode']!='projection_only_v1' or m.get('align_branch_control')):
            raise ValueError('Wrong baseline source/mode/budget/selection')
        if (t['lambda_rank_align']!=.03 or t['lambda_srocc_align']!=.04 or t['lambda_align_aux']!=.3 or
                t['layerwise_lr']['align_mult']!=1. or t['layerwise_lr']['shared_projection_mult']!=1. or
                t['lambda_srocc_quality']!={'agiqa3k':.02,'aigciqa2023':.04}[ds]):
            raise ValueError('Restore baseline LR/loss weights; keep both branch auxiliary losses')
        n=m['nss_text_gate']
        if n.get('descriptor','legacy')!='legacy' or n['nss_dim']!=6 or n.get('pathway','both')!='both':
            raise ValueError('Original six-dimensional NSS required')
        name=f'{ds}_projection_only_v1_v2_{v}_train42_split42'
        c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
        m['align_branch_control']=copy.deepcopy(r.VARIANTS[v])
        t.update(record_split_manifest=True,record_optimizer_initialization=True)
        return c

    original_verify=r.verify
    def verify(v,ds,split,source):
        original_verify(v,ds,split,source)
        import csv,json,math
        c=config(v,ds,split);_,table=r.paths(v,ds);ctrl=c['model']['align_branch_control']
        expected={
            'global_output':float(ctrl['output']=='global'),
            'half_mix':float(ctrl['output']=='half'),
            'uniform_pool':float(ctrl['local_pool']=='uniform'),
            'gate_trainable':float(ctrl['output']=='learned'),
            'match_trainable':float(ctrl['local_pool']=='text'),
        }
        with table.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            if row['event'] not in ('epoch','best_val','best_test'):continue
            m=json.loads(row['metrics_json']);prefix='val_val' if row['event']=='epoch' else ('val' if row['event']=='best_val' else 'test')
            for key,value in expected.items():
                if m.get(prefix+'_align_control_'+key)!=value:
                    raise ValueError('Actual alignment control differs: '+key)
            gate=1. if ctrl['output']=='global' else .5 if ctrl['output']=='half' else None
            if gate is not None and not math.isclose(m[prefix+'_align_gate'],gate,abs_tol=1e-7):
                raise ValueError('Constant mixing coefficient differs')
    r.config,r.verify=config,verify
    return r

if __name__=='__main__':make_runner().main()
