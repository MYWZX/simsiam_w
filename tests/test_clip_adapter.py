"""Post-CLIP adapter: exact old defaults, live gradients, strict transfer and source pipelines."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
from unittest.mock import patch

import torch
import yaml
from models.two_stage_models import GlobalResidualAdapter,load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state,build_optimizer as preopt
from train_cm_iqa import build_optimizer as downopt
from test_frozen_clip_baseline import small_model
from test_projection_only import config as tiny_config
import test_pretrain_component_strength as fixture_tests
from scripts import run_clip_adapter as sweep


def config(enabled=True,frozen=True):
    c=tiny_config(True);c['model']['global_encoder']['freeze']=frozen
    if enabled:c['model']['global_adapter']={'enabled':True,'hidden_dim':4,'residual_scale':.1}
    c['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'global_adapter_mult':1.,'fusion_mult':.1,'shared_projection_mult':1.})
    return c


class AdapterModelTests(unittest.TestCase):
    def setUp(self):
        rng=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,rng);torch.manual_seed(42)

    def test_identity_initialization_preserves_existing_weights_rng_and_eval_outputs(self):
        c=config(False);torch.manual_seed(42);old=small_model(c).eval()
        torch.manual_seed(42);new=small_model(config()).eval()
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(4,3,16,16);a=old.forward_single(x,['cat']*4);b=new.forward_single(x,['cat']*4)
        for k in ['f_sem','f_loc','s_quality','s_align']:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        self.assertFalse(any('global_adapter' in k for k in old.state_dict()))
        self.assertTrue(any('global_adapter' in k for k in new.state_dict()))

    def test_frozen_or_unfrozen_visual_adapter_always_learns_and_local_unchanged(self):
        for frozen in [True,False]:
            m=small_model(config(frozen=frozen));m.train();a=m.backbone.global_adapter
            self.assertTrue(a.training);self.assertEqual(m.backbone.global_encoder.training,not frozen)
            before={k:v.clone() for k,v in m.backbone.global_encoder.state_dict().items()}
            opt=downopt(config(frozen=frozen),m)
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                out=m.forward_single(torch.randn(4,3,16,16),['cat','dog','bird','fish'])
                from utils.downstream_controls import audit_fusion_gradients
                audit=audit_fusion_gradients(m,{'quality':out['s_quality'].sum(),'alignment':out['s_align'].sum()})
                self.assertEqual(audit['gradient_audit_global_adapter_present'],1.)
                self.assertGreater(audit['gradient_global_adapter_quality_norm'],0.)
                self.assertGreater(audit['gradient_global_adapter_alignment_norm'],0.)
                self.assertEqual(audit['gradient_audit_cross_attention_present'],0.)
                (out['s_quality'].square().sum()+out['s_align'].square().sum()).backward()
                self.assertGreater(a.up.weight.grad.abs().sum().item(),0.)
                if step:self.assertGreater(a.down.weight.grad.abs().sum().item(),0.)
                for p in m.backbone.global_encoder.parameters():
                    if frozen:self.assertIsNone(p.grad)
                    else:self.assertIsNotNone(p.grad)
                self.assertGreater(m.local_patch_proj.weight.grad.abs().sum().item(),0.)
                opt.step()
            if frozen:
                for k,v in m.backbone.global_encoder.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            self.assertTrue(all(p.requires_grad for p in a.parameters()))

    def test_adapter_receives_quality_alignment_and_all_three_views(self):
        c=config();m=small_model(c,joint=True).eval();calls=[]
        h=m.iqa.backbone.global_adapter.register_forward_hook(lambda mod,ins,out:calls.append(out))
        b=m._forward_three_views(*[torch.randn(4,3,16,16) for _ in range(3)],['cat','dog','bird','fish'],['dog','bird','fish','cat']);h.remove()
        self.assertEqual(len(calls),3)
        w=m.iqa.backbone.global_adapter.up.weight
        for loss in [b['quality_pred_clean'].sum(),b['align_pred_clean'].sum()]:
            self.assertGreater(torch.autograd.grad(loss,w,retain_graph=True)[0].abs().sum().item(),0.)
        from train_cm_pretrain import _quality_view_simsiam_loss
        self.assertGreater(torch.autograd.grad(_quality_view_simsiam_loss(b,torch.ones(4)),w)[0].abs().sum().item(),0.)

    def test_optimizer_new_group_only_and_all_existing_groups_unchanged(self):
        for joint,build,stage in [(True,preopt,'pretrain'),(False,downopt,'agiqa3k')]:
            real=sweep.expected_config('frozen',stage);c=config(False)
            for k in ['lr','weight_decay','layerwise_lr']:
                if k in real['train']:c['train'][k]=copy.deepcopy(real['train'][k])
            old=small_model(c,joint);enabled=copy.deepcopy(c);enabled['model']['global_adapter']={'enabled':True,'hidden_dim':4,'residual_scale':.1};new=small_model(enabled,joint)
            def groups(model):
                names={id(p):n for n,p in model.named_parameters()}
                return {g['name']:(g['lr'],g['weight_decay'],[names[id(p)] for p in g['params']]) for g in build(c,model).param_groups}
            before=groups(old);after=groups(new);adapter=after.pop('global_adapter')
            self.assertEqual(before,after);self.assertAlmostEqual(adapter[0],1e-4)
            self.assertTrue(all('backbone.global_adapter.' in k for k in adapter[2]))
            opt=build(c,new);iqa=new.iqa if joint else new
            group=next(g for g in opt.param_groups if any(p is iqa.backbone.fusion.semantic_proj.weight for p in g['params']))
            self.assertAlmostEqual(group['lr'],1e-5)

    def test_complete_transfer_mismatch_partial_shape_scale_rejected(self):
        source=small_model(config(),joint=True);state=_extract_representation_state(source.state_dict())
        keys=[k for k in state if 'backbone.global_adapter.' in k];self.assertEqual(len(keys),7)
        target=small_model(config(frozen=False))
        def load(target,s):
            with patch('torch.load',return_value={'model':s}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        load(target,state)
        for k,v in source.iqa.backbone.global_adapter.state_dict().items():torch.testing.assert_close(v,target.backbone.global_adapter.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(all(p.requires_grad for p in target.backbone.global_encoder.parameters()))
        for k in keys:
            bad=dict(state);bad.pop(k)
            with self.assertRaisesRegex(RuntimeError,'adapter'):load(target,bad)
        missing_clip=dict(state);missing_clip.pop(next(k for k in state if k.startswith('iqa.backbone.global_encoder.')))
        with self.assertRaisesRegex(RuntimeError,'CLIP'):load(target,missing_clip)
        with self.assertRaisesRegex(RuntimeError,'adapter'):load(small_model(config(False)),state)
        old=small_model(config(False),joint=True)
        with self.assertRaisesRegex(RuntimeError,'adapter'):load(target,_extract_representation_state(old.state_dict()))
        for key,value in [('hidden_dim',5),('residual_scale',.2)]:
            c=config();c['model']['global_adapter'][key]=value
            with self.assertRaisesRegex(RuntimeError,'adapter'):load(small_model(c),state)

    def test_invalid_configs_rejected(self):
        for hidden,scale in [(0,.1),(4,0),(4,float('nan'))]:
            with self.assertRaises(ValueError):GlobalResidualAdapter(8,hidden,scale)
        c=config();c['model']['fusion']['mode']='attention'
        with self.assertRaisesRegex(ValueError,'projection_only'):small_model(c)


class AdapterPipelineTests(unittest.TestCase):
    def setUp(self):
        rng=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,rng)
        self.fixture=fixture_tests.ComponentStrengthTests('test_incomplete_refuses_pack');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.root=self.fixture.root
        shutil.copytree(sweep.ROOT/sweep.CONFIG,self.root/sweep.CONFIG)
        p=patch.object(sweep,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def fake_source(self):
        r=sweep.source_runner();Base=self.fixture.fake_trainer(r)
        class Process(Base):
            def __init__(inner,*args,**kwargs):
                super().__init__(*args,**kwargs)
                p=r.paths('pretrain')[0]/'best_component_anchor_pretrain.pth';s=torch.load(p,weights_only=False)
                s['model'].update({'iqa.backbone.global_adapter.'+k:v for k,v in GlobalResidualAdapter(768,256,.1).state_dict().items()});torch.save(s,p)
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Process),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        return r,r.completed('pretrain')

    def test_configs_source_sharing_and_cpu_dry_guard(self):
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v)
            for stage in sweep.STAGES:
                c=r.validate(stage)
                expected=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain')) if stage=='pretrain' else yaml.safe_load((self.root/f'configs/current_tuning_reference/{stage}.yaml').read_text())
                expected['model']['global_encoder']['freeze']=True if stage=='pretrain' else v=='frozen'
                expected['model']['global_adapter']=copy.deepcopy(sweep.ADAPTER)
                expected['train']['layerwise_lr']['global_adapter_mult']=1e-4/expected['train']['lr']
                for k in ['run_id','output_dir','results_csv']:expected[k]=c[k]
                if stage!='pretrain':expected['train']['pretrained_checkpoint']=r.validate('pretrain')['output_dir']+'/best_component_anchor_pretrain.pth'
                self.assertEqual(c,expected)
            with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('pretrain')
            with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):sweep.pipeline.run_pipeline(r,'continue',True)
            self.assertFalse(r.paths('pretrain')[0].exists())
        self.assertEqual(sweep.make_runner('frozen').paths('pretrain'),sweep.make_runner('unfrozen').paths('pretrain'))
        self.assertNotEqual(sweep.make_runner('frozen').paths('agiqa1k'),sweep.make_runner('unfrozen').paths('agiqa1k'))
        with self.assertRaises(ValueError):sweep.make_runner('unknown')

    def test_six_downstream_sources_pack_and_no_overwrite(self):
        from utils.agiqa1k_subset1078 import expected_resplit_manifest
        src,source=self.fake_source()
        checkpoint=Path(source['checkpoint']);broken=torch.load(checkpoint,weights_only=False)
        broken['model'].pop('iqa.backbone.global_adapter.norm.bias')
        bad=self.root/'incomplete_adapter.pth';torch.save(broken,bad)
        with self.assertRaisesRegex(ValueError,'global adapter'):src.verify_checkpoint(bad,src.validate('pretrain'),4)
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v);self.assertEqual(r.completed('pretrain'),source)
            class Process:
                def __init__(inner,cmd,**kwargs):
                    stage=Path(cmd[3]).stem;c=r.validate(stage);d,table=r.paths(stage)
                    (d/'config_snapshot.yaml').write_text(yaml.safe_dump(c));r.write_new(d/'run_metadata.json',dict(seed=42,split_seed=c['split_seed'],checkpoint_verified_after_load=True,checkpoint={'sha256':source['sha256']}))
                    if c['train'].get('record_split_manifest',False):r.write_new(d/'data_split_manifest.json',dict(train_seed=42,split_seed=c['split_seed'],indices=expected_resplit_manifest(3407)['split_indices'] if stage=='agiqa1k' else {}))
                    with table.open('w') as f:
                        w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                        for e in range(1,61):w.writerow(dict(event='epoch',epoch=e,metrics_json='{}'))
                        w.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
                    inner.stdout=iter(['fake downstream\n'])
                def wait(inner):return 0
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Process),patch.object(r.subprocess,'run'),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                for stage in sweep.STAGES[1:]:r.run(stage);self.assertEqual(r.completed(stage),source)
                r.pack('downstream')
                with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('already complete')):sweep.pipeline.run_pipeline(r,'continue')
            with tarfile.open(self.root/f'{r.VERSION}_downstream_results.tar.gz') as t:
                self.assertEqual(sum(n.endswith('.csv') for n in t.getnames()),4);self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in t.getnames()))
            with self.assertRaises(FileExistsError):r.run('pretrain')
        # An adapterless/corrupt checkpoint must never be usable as this experiment's source.
        d,_=sweep.source_runner().paths('pretrain');(d/'best_component_anchor_pretrain.pth').write_bytes(b'old checkpoint')
        with self.assertRaises(ValueError):sweep.make_runner('frozen').completed('pretrain')


if __name__=='__main__':unittest.main()
