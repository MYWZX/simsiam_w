"""Pairwise alignment-rank weights 0.015/0.06; four original-source downstream runs."""
import importlib.util
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={'arank050':.015,'arank200':.06}


def make_runner():
    # Each protocol owns its globals; never mutate the previous LR/SROCC study.
    spec=importlib.util.spec_from_file_location('_align_pairwise_rank_protocol',ROOT/'scripts/run_align_tuning.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.__doc__=__doc__;r.ROOT=ROOT
    r.VARIANTS=dict(VARIANTS);r.QUEUES={'rank':tuple(VARIANTS),'low':('arank050',),'high':('arank200',)}
    r.VERSION='projection_only_v1_v2_align_rank';r.PROTOCOL_DOC='docs/align_rank.md'

    def config(variant,ds,split=None):
        if variant not in r.VARIANTS or ds not in r.DATASETS:
            raise ValueError('Unknown alignment-rank variant/dataset; 1K is excluded')
        if split is not None and split!=42:raise ValueError('Fixed train/split seed42')
        c=yaml.safe_load((r.ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())
        t=c['train'];m=c['model'];reference=r.reference()
        checkpoint='outputs/'+reference['source']['checkpoint'].split('/outputs/',1)[1]
        if (c['seed']!=42 or c['split_seed']!=42 or t['epochs']!=60 or
                t['selection_metric']!='val_quality_SROCC' or t['selection_mode']!='max' or
                t['pretrained_checkpoint']!=checkpoint or t['pretrained_load_scope']!='representation' or
                m['fusion']['mode']!='projection_only_v1'):
            raise ValueError('Wrong pinned baseline source/mode/budget/selection')
        n=m['nss_text_gate']
        if n.get('descriptor','legacy')!='legacy' or n['nss_dim']!=6 or n.get('pathway','both')!='both':
            raise ValueError('Original six-dimensional NSS required')
        if (t['lambda_rank_align']!=.03 or t['lambda_srocc_align']!=.04 or t['srocc_tau']!=.02 or
                t['layerwise_lr']['align_mult']!=1. or t['layerwise_lr']['shared_projection_mult']!=1. or
                t['lambda_srocc_quality']!={'agiqa3k':.02,'aigciqa2023':.04}[ds]):
            raise ValueError('Restore original baseline LR and loss weights before rank tuning')
        name=f'{ds}_projection_only_v1_v2_{variant}_train42_split42'
        c.update(run_id=name,output_dir=f'outputs/{name}',results_csv=f'outputs/{name}.csv')
        t.update(lambda_rank_align=r.VARIANTS[variant],record_split_manifest=True,record_optimizer_initialization=True)
        return c

    r.config=config
    return r

if __name__=='__main__':make_runner().main()
