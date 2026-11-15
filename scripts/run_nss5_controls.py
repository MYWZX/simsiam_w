"""Five-stat cleanup: normalization control and fixed-normalization pathway ablations."""
import argparse,copy,os,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_pretrain_component_strength as pipeline
from models.nss import MODES
VARIANTS={'legacy_both':('legacy_stats5_v1','both'),
          'mscn_both':('mscn_stats5_v1','both'),
          'legacy_inject':('legacy_stats5_v1','inject_only'),
          'legacy_gate':('legacy_stats5_v1','gate_only')}
CONFIG=Path('configs/nss5_controls');STAGES=pipeline.STAGES


def expected_config(variant,stage='pretrain'):
    if variant not in VARIANTS or stage not in STAGES:raise ValueError('Unknown five-stat variant/stage')
    if stage=='pretrain':
        cfg=copy.deepcopy(pipeline.baseline.validate(stage))
        run_id=f'richhf18k_projection_only_v1_ctx5_nss5_{variant}_seed42'
    else:
        cfg=yaml.safe_load((ROOT/f'configs/current_tuning_reference/{stage}.yaml').read_text())
        prefix='agiqa1k_subset1078' if stage=='agiqa1k' else stage
        run_id=f'{prefix}_projection_only_v1_nss5_{variant}_train42_split{cfg["split_seed"]}'
        cfg['train']['pretrained_checkpoint']=expected_config(variant)['output_dir']+'/best_component_anchor_pretrain.pth'
    mode,pathway=VARIANTS[variant]
    cfg['model']['nss_text_gate'].update(descriptor=mode,nss_dim=5,pathway=pathway)
    cfg.update(run_id=run_id,output_dir=f'outputs/{run_id}',results_csv=f'outputs/{run_id}.csv')
    return cfg


def make_runner(variant):
    expected_config(variant)
    r=pipeline.make_runner(variant,config_factory=expected_config,config_dir=CONFIG,
                          version=f'projection_only_v1_ctx5_nss5_{variant}')
    original=r.verify_checkpoint
    def verify(path,cfg,epoch):
        original(path,cfg,epoch)
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        mode,pathway=VARIANTS[variant]
        if not torch.equal(state.get('iqa.nss_descriptor_signature',torch.empty(0)).cpu(),torch.tensor(MODES[mode])):
            raise ValueError('Five-stat NSS descriptor signature mismatch')
        if not torch.equal(state.get('iqa.nss_pathway_signature',torch.empty(0)).cpu(),torch.tensor([{'both':1,'inject_only':2,'gate_only':3}[pathway]])):
            raise ValueError('Five-stat NSS pathway signature mismatch')
        h=int(cfg['model']['nss_text_gate'].get('gate_hidden_dim',cfg['model']['heads'].get('mlp_hidden_dim',256)))
        e=int(cfg['model']['fusion']['embed_dim']);gate_in=e if pathway=='inject_only' else e+5
        shapes={'nss_gate.0.weight':(h,gate_in),'nss_gate.0.bias':(h,),'nss_gate.3.weight':(1,h),'nss_gate.3.bias':(1,)}
        if pathway!='gate_only':shapes.update({'nss_proj.0.weight':(h,5),'nss_proj.0.bias':(h,),'nss_proj.2.weight':(e,h),'nss_proj.2.bias':(e,)})
        found={k.removeprefix('iqa.'):v for k,v in state.items() if k.startswith(('iqa.nss_proj.','iqa.nss_gate.'))}
        if set(found)!=set(shapes) or any(tuple(found[k].shape)!=shape for k,shape in shapes.items()):
            raise ValueError('Incomplete/mismatched five-stat NSS source tensors')
    r.verify_checkpoint=verify
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('variant',choices=VARIANTS)
    p.add_argument('action',nargs='?',default='pretrain',choices=STAGES+('all','downstream','continue','pack','pack-downstream'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();os.chdir(ROOT);r=make_runner(a.variant)
    if a.action.startswith('pack'):
        if a.dry_run:p.error('--dry-run applies to run actions')
        r.pack('pretrain' if a.action=='pack' else 'downstream')
    elif a.action in ('all','continue','downstream'):
        pipeline.run_pipeline(r,'continue' if a.action=='all' else a.action,a.dry_run)
    else:pipeline.run_stage(r,a.action,a.dry_run)

if __name__=='__main__':main()
