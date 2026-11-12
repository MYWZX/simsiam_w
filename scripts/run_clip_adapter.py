"""Shared frozen-CLIP + residual-adapter pretraining, then frozen/unfrozen downstream."""
import argparse
import copy
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_pretrain_component_strength as pipeline

CONFIG = Path('configs/clip_adapter')
VARIANTS = ('frozen', 'unfrozen')
STAGES = pipeline.STAGES
SOURCE_VERSION = 'projection_only_v1_ctx5_clip_adapter_v1'
ADAPTER = {'enabled': True, 'hidden_dim': 256, 'residual_scale': 0.1}


def expected_config(variant, stage='pretrain'):
    if variant not in VARIANTS+('source',) or stage not in STAGES:
        raise ValueError('Unknown CLIP adapter condition/stage')
    if stage == 'pretrain':
        cfg=copy.deepcopy(pipeline.baseline.validate(stage))
        run_id=f'richhf18k_{SOURCE_VERSION}_seed42'
        cfg['model']['global_encoder']['freeze']=True
    else:
        if variant == 'source':raise ValueError('source has only a pretraining stage')
        cfg=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        cfg['model']['global_encoder']['freeze']=variant=='frozen'
        prefix='agiqa1k_subset1078' if stage=='agiqa1k' else stage
        run_id=f'{prefix}_projection_only_v1_clip_adapter_v1_{variant}_train42_split{cfg["split_seed"]}'
        cfg['train']['pretrained_checkpoint']=expected_config('source')['output_dir']+'/best_component_anchor_pretrain.pth'
    cfg['model']['global_adapter']=copy.deepcopy(ADAPTER)
    # Dedicated trainable group at 1e-4 in both stages; all existing groups stay unchanged.
    cfg['train']['layerwise_lr']['global_adapter_mult']=1e-4/cfg['train']['lr']
    cfg.update(run_id=run_id,output_dir=f'outputs/{run_id}',results_csv=f'outputs/{run_id}.csv')
    return cfg


def source_runner():
    r=pipeline.make_runner('source',config_factory=expected_config,config_dir=CONFIG,version=SOURCE_VERSION)
    original=r.verify_checkpoint
    def verify(path,cfg,epoch):
        original(path,cfg,epoch)
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        prefix='iqa.backbone.global_adapter.'
        shapes={'signature':(3,),'norm.weight':(768,),'norm.bias':(768,),
                'down.weight':(256,768),'down.bias':(256,),'up.weight':(768,256),'up.bias':(768,)}
        found={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if set(found)!=set(shapes) or any(tuple(found[k].shape)!=shape for k,shape in shapes.items()):
            raise ValueError('Incomplete/mismatched global adapter source checkpoint')
        if not torch.equal(found['signature'].cpu(),torch.tensor([768,256,.1],dtype=torch.float64)):
            raise ValueError('Global adapter source signature mismatch')
    r.verify_checkpoint=verify
    return r


def make_runner(variant):
    if variant not in VARIANTS:raise ValueError('Unknown CLIP adapter condition')
    r=pipeline.make_runner(variant,config_factory=expected_config,config_dir=CONFIG,
                          version=SOURCE_VERSION+'_'+variant)
    source=source_runner();completed=r.completed;run=r.run
    def checked(stage):return source.completed(stage) if stage=='pretrain' else completed(stage)
    def launch(stage,dry_run=False):return source.run(stage,dry_run) if stage=='pretrain' else run(stage,dry_run)
    r.completed,r.run=checked,launch
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=VARIANTS)
    p.add_argument('action',nargs='?',default='pretrain',choices=STAGES+('all','downstream','continue','pack','pack-downstream'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT)
    r=make_runner(a.variant)
    if a.action.startswith('pack'):
        if a.dry_run:p.error('--dry-run applies only to runs')
        r.pack('pretrain' if a.action=='pack' else 'downstream')
    elif a.action in ('all','downstream','continue'):
        pipeline.run_pipeline(r,'continue' if a.action=='all' else a.action,a.dry_run)
    else:pipeline.run_stage(r,a.action,a.dry_run)


if __name__=='__main__':main()
