"""Ten fresh single-factor pretraining experiments from the original six-channel baseline."""
import argparse,copy,os,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_pretrain_component_strength as pipeline
VARIANTS={
 'answer010':(('lambda_img_answer',),.10), 'answer030':(('lambda_img_answer',),.30),
 'pa025':(('lambda_prompt_answer',),.025), 'pa100':(('lambda_prompt_answer',),.10),
 'local025':(('lambda_img_prompt_local',),.025), 'local100':(('lambda_img_prompt_local',),.10),
 'word025':(('lambda_prompt_answer_word',),.025), 'word100':(('lambda_prompt_answer_word',),.10),
 'anchortau010':(('quality_component_anchor_ordinal','anchor_temperature'),.10),
 'anchortau030':(('quality_component_anchor_ordinal','anchor_temperature'),.30),
}
QUEUES={'low':tuple(VARIANTS)[::2],'high':tuple(VARIANTS)[1::2]}
CONFIG=Path('configs/pretrain_objectives');STAGES=pipeline.STAGES
ACTIONS=STAGES+('all','downstream','continue','pack','pack-downstream')


def expected_config(variant,stage='pretrain'):
    if variant not in VARIANTS or stage not in STAGES:raise ValueError('Unknown pretraining factor variant/stage')
    if stage=='pretrain':
        cfg=copy.deepcopy(pipeline.baseline.validate(stage))
        run_id=f'richhf18k_projection_only_v1_ctx5_obj_{variant}_seed42'
        path,value=VARIANTS[variant];target=cfg['train']
        for key in path[:-1]:target=target[key]
        target[path[-1]]=value
    else:
        cfg=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix='agiqa1k_subset1078' if stage=='agiqa1k' else stage
        run_id=f'{prefix}_projection_only_v1_obj_{variant}_train42_split{cfg["split_seed"]}'
        cfg['train']['pretrained_checkpoint']=expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
    cfg.update(run_id=run_id,output_dir=f'outputs/{run_id}',results_csv=f'outputs/{run_id}.csv')
    return cfg


def make_runner(variant):
    return pipeline.make_runner(variant,config_factory=expected_config,config_dir=CONFIG,
                                version=f'projection_only_v1_ctx5_obj_{variant}')


def dispatch(r,action,dry_run=False):
    if action.startswith('pack'):
        if dry_run:raise ValueError('--dry-run applies only to run actions')
        r.pack('pretrain' if action=='pack' else 'downstream')
    elif action in ('all','continue','downstream'):
        pipeline.run_pipeline(r,'continue' if action=='all' else action,dry_run)
    else:pipeline.run_stage(r,action,dry_run)


def run_queue(queue,action='pretrain',dry_run=False):
    if queue not in QUEUES or action not in ('pretrain','all','downstream','continue','pack','pack-downstream'):
        raise ValueError('Queues accept pretrain/all/downstream/continue/pack/pack-downstream')
    runners=[make_runner(v) for v in QUEUES[queue]]
    # Validate every configuration/output before an expensive queue starts.
    for r in runners:
        if action.startswith('pack'):
            if dry_run:raise ValueError('--dry-run applies only to run actions')
            stages=STAGES if action=='pack-downstream' else ('pretrain',)
            for stage in stages:r.completed(stage)
            archive=ROOT/f'{r.VERSION}_{"downstream" if action=="pack-downstream" else "pretrain"}_results.tar.gz'
            if archive.exists():raise FileExistsError(f'Preserve existing archive: {archive}')
        elif action=='pretrain':
            d,table=r.paths('pretrain')
            if d.exists() or table.exists():r.completed('pretrain')
            else:r.run('pretrain',True)
        else:pipeline.run_pipeline(r,'continue' if action=='all' else action,True)
    if dry_run:return
    for r in runners:
        if action=='pretrain' and any(p.exists() for p in r.paths('pretrain')):
            r.completed('pretrain');print('Verified completed pretraining; skip:',r.VERSION);continue
        dispatch(r,action)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=tuple(VARIANTS)+tuple(QUEUES)+('check-configs',))
    p.add_argument('action',nargs='?',default='pretrain',choices=ACTIONS)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT)
    if a.variant=='check-configs':
        if a.action!='pretrain' or a.dry_run:p.error('check-configs takes no other arguments')
        for v in VARIANTS:
            r=make_runner(v)
            for stage in STAGES:r.validate(stage)
        print('Validated 10 pretrain + 30 downstream configurations; five independent factors, original six-channel baseline')
    elif a.variant in QUEUES:run_queue(a.variant,a.action,a.dry_run)
    else:dispatch(make_runner(a.variant),a.action,a.dry_run)

if __name__=='__main__':main()
