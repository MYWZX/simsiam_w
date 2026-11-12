"""Fresh standard-MSCN ablations; optional three-dataset downstream pipeline."""
import argparse
import copy
import os
from pathlib import Path
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_pretrain_component_strength as pipeline
from models.nss import MODES
VARIANTS = {'stats6':'mscn_stats6_v1','ggd':'mscn_ggd_v1'}
CONFIG = Path('configs/nss_ablation')
STAGES = pipeline.STAGES


def expected_config(variant,stage='pretrain'):
    if variant not in VARIANTS or stage not in STAGES:
        raise ValueError('Unknown NSS variant or stage')
    if stage=='pretrain':
        cfg=copy.deepcopy(pipeline.baseline.validate(stage))
        run_id=f'richhf18k_projection_only_v1_ctx5_{VARIANTS[variant]}_seed42'
    else:
        cfg=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix='agiqa1k_subset1078' if stage=='agiqa1k' else stage
        run_id=f'{prefix}_projection_only_v1_{VARIANTS[variant]}_train42_split{cfg["split_seed"]}'
        cfg['train']['pretrained_checkpoint']=expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
    cfg['model']['nss_text_gate'].update(descriptor=VARIANTS[variant],nss_dim=MODES[VARIANTS[variant]][1])
    cfg.update(run_id=run_id,output_dir=f'outputs/{run_id}',results_csv=f'outputs/{run_id}.csv')
    return cfg


def make_runner(variant):
    expected_config(variant)
    runner=pipeline.make_runner(variant,config_factory=expected_config,config_dir=CONFIG,
                                version=f'projection_only_v1_ctx5_{VARIANTS[variant]}')
    original=runner.verify_checkpoint
    def verify_checkpoint(path,cfg,epoch):
        original(path,cfg,epoch)
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        code,dim=MODES[VARIANTS[variant]]
        signature=state.get('iqa.nss_descriptor_signature')
        if signature is None or not torch.equal(signature.cpu(),torch.tensor([code,dim])):
            raise ValueError('NSS checkpoint descriptor signature mismatch')
        hidden=int(cfg['model']['nss_text_gate'].get('gate_hidden_dim',cfg['model']['heads'].get('mlp_hidden_dim',256)))
        fused=int(cfg['model']['fusion']['embed_dim'])
        shapes={'nss_proj.0.weight':(hidden,dim),'nss_proj.0.bias':(hidden,),
                'nss_proj.2.weight':(fused,hidden),'nss_proj.2.bias':(fused,),
                'nss_gate.0.weight':(hidden,fused+dim),'nss_gate.0.bias':(hidden,),
                'nss_gate.3.weight':(1,hidden),'nss_gate.3.bias':(1,)}
        for key,shape in shapes.items():
            value=state.get('iqa.'+key)
            if value is None or tuple(value.shape)!=shape:
                raise ValueError(f'Incomplete NSS source tensor: {key}')
    runner.verify_checkpoint=verify_checkpoint
    return runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=VARIANTS)
    p.add_argument('action',nargs='?',default='pretrain',choices=STAGES+('all','downstream','continue','pack','pack-downstream'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT)
    r=make_runner(a.variant)
    if a.action.startswith('pack'):
        if a.dry_run:p.error('--dry-run is for run actions')
        r.pack('pretrain' if a.action=='pack' else 'downstream')
    elif a.action in ('all','continue','downstream'):
        pipeline.run_pipeline(r,'continue' if a.action=='all' else a.action,a.dry_run)
    else:pipeline.run_stage(r,a.action,a.dry_run)

if __name__=='__main__':main()
