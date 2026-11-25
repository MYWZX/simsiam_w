"""Fresh legacy-normalized kurtosis NSS source and its three downstream controls."""
import argparse
import copy
import os
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_pretrain_component_strength as pipeline
VARIANT='kurt6'
CONFIG=Path('configs/nss_kurtosis')


def expected_config(variant,stage='pretrain'):
    if variant!=VARIANT or stage not in pipeline.STAGES:
        raise ValueError('Unknown kurtosis stage')
    if stage=='pretrain':
        c=copy.deepcopy(pipeline.baseline.validate('pretrain'))
        name='richhf18k_projection_only_v1_ctx5_legacy_kurt6_v1_seed42'
    else:
        c=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix='agiqa1k_subset1078' if stage=='agiqa1k' else stage
        name=f'{prefix}_projection_only_v1_legacy_kurt6_v1_train42_split{c["split_seed"]}'
        c['train']['pretrained_checkpoint']=expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
        c['train'].update(record_split_manifest=True,record_optimizer_initialization=True)
    c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
    c['model']['nss_text_gate'].update(descriptor='legacy_kurt6_v1',nss_dim=6,pathway='both')
    return c


def make_runner():
    r=pipeline.make_runner(VARIANT,config_factory=expected_config,config_dir=CONFIG,
                           version='projection_only_v1_ctx5_legacy_kurt6_v1')
    original=r.verify_checkpoint
    def verify(path,cfg,epoch):
        original(path,cfg,epoch)
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        if not torch.equal(state.get('iqa.nss_descriptor_signature',torch.empty(0)),torch.tensor([5,6])):
            raise ValueError('Missing kurtosis NSS signature')
        for name in ('nss_proj.0.weight','nss_proj.0.bias','nss_proj.2.weight','nss_proj.2.bias',
                     'nss_gate.0.weight','nss_gate.0.bias','nss_gate.3.weight','nss_gate.3.bias'):
            if 'iqa.'+name not in state:raise ValueError('Incomplete kurtosis NSS source: '+name)
    r.verify_checkpoint=verify
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',default='pretrain',choices=pipeline.STAGES+('all','continue','downstream','pack','pack-downstream'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT);r=make_runner()
    if a.action.startswith('pack'):
        if a.dry_run:p.error('Cannot dry-run packing')
        r.pack('pretrain' if a.action=='pack' else 'downstream')
    elif a.action in ('all','continue','downstream'):
        pipeline.run_pipeline(r,a.action,a.dry_run)
    else:pipeline.run_stage(r,a.action,a.dry_run)
if __name__=='__main__':main()
