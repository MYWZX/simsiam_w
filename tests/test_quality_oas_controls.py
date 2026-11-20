"""Synthetic forwards/gradients and artifact protocol only; no real training."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from scripts import run_quality_oas_controls as sweep
from utils import pretrain_phases as phases
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import (train_one_epoch, build_optimizer, _extract_representation_state,
    _compute_quality_component_anchor_ordinal_losses)
from test_dual_quality_representation import config, batch, views
from test_projection_ablations import make
from test_quality_route_ablations import Global, Local, Text
import test_pretrain_component_strength as fixtures


def tiny(phase='joint',residual=False):
    c=config();c.update(seed=42,split_seed=42,run_id='quality_test')
    c['model']['fusion']['mode']='projection_only_v1'
    c['model'].update(shared_simsiam_projection=True,image_image_feature='g',image_text_feature='g',shared_projection_project_patches=False)
    c['model']['quality_adapter']['mode']='residual_feature'
    if residual:c['model']['dual_quality_representation']['anchor_global_source']='residual'
    t=c['train'];t.update(pretraining_phase=phase,pretrain_selection_mode='max')
    if phase!='joint':
        t['pretrain_selection_metric']='val_component_anchor_SROCC' if phase=='quality' else 'val_align_proxy_SROCC'
        t['pcgrad']={'enabled':False}
        for k in (phases.ALIGN_LOSSES if phase=='quality' else phases.QUALITY_LOSSES):
            if k in t:t[k]=0.
        if phase=='alignment':
            t['quality_auxiliary_supervision']={'enabled':False}
            t.update(quality_stage_checkpoint='fake.pth',quality_stage_run_id='quality_test')
    return c


class ModelTests(unittest.TestCase):
    def setUp(self):
        state=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,state);torch.manual_seed(42)

    def test_oas_uses_actual_post_add_tensor_and_only_changes_component_supervision(self):
        base=make(tiny());changed=make(tiny(residual=True));changed.load_state_dict(base.state_dict(),strict=True)
        b=batch();a=views(base,b);seen=[]
        hook=changed.iqa.global_quality_head.register_forward_pre_hook(lambda m,ins:seen.append(ins[0]))
        z=views(changed,b);hook.remove()
        # First global-head call is the real residual used for Clean scoring;
        # later calls are existing diagnostics on raw/adapted features.
        h=seen[0];rho=torch.sigmoid(changed.iqa.quality_feature_mix_logit)
        torch.testing.assert_close(z['component_anchor_clean_z'],torch.nn.functional.normalize((1-rho)*h+rho*z['quality_local_clean_z'],dim=-1))
        for k in a:
            torch.testing.assert_close(a[k],z[k],rtol=0,atol=0)
        losses,pred=_compute_quality_component_anchor_ordinal_losses(z,b,torch.ones(3),tiny()['train'])
        expected={**z,'quality_clean_z':z['component_anchor_clean_z']};expected.pop('component_anchor_clean_z')
        other,op=_compute_quality_component_anchor_ordinal_losses(expected,b,torch.ones(3),tiny()['train'])
        for k in pred:torch.testing.assert_close(pred[k],op[k])
        total=sum(losses[k] for k in losses if k.endswith(('anchor_ce','ordinal_bce')))
        g=torch.autograd.grad(total,changed.iqa.quality_adapter_gate_logit)[0]
        self.assertTrue(torch.isfinite(g));self.assertGreater(g.abs().item(),0)
        oldloss,_=_compute_quality_component_anchor_ordinal_losses(a,b,torch.ones(3),tiny()['train'])
        oldtotal=sum(v for k,v in oldloss.items() if k.endswith(('anchor_ce','ordinal_bce')))
        self.assertIsNone(torch.autograd.grad(oldtotal,base.iqa.quality_adapter_gate_logit,allow_unused=True)[0])

    def test_oas_has_no_new_weights_rng_draws_and_strict_transfer_matches_downstream(self):
        torch.manual_seed(7);old=make(tiny());rng=torch.random.get_rng_state()
        torch.manual_seed(7);new=make(tiny(residual=True));self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(old.state_dict().keys(),new.state_dict().keys())
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
        target=make(tiny(),joint=False);state=_extract_representation_state(new.state_dict())
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(target,'fake.pth',load_scope='representation')
        for k,v in state.items():torch.testing.assert_close(v,target.state_dict()[k.removeprefix('iqa.')],rtol=0,atol=0)
        b=batch();a=old.iqa.forward_single(b['image_clean'],b['prompt']);z=new.iqa.forward_single(b['image_clean'],b['prompt'])
        for k in ('s_quality','s_align','quality_representation'):torch.testing.assert_close(a[k],z[k],rtol=0,atol=0)

    def test_unsupported_oas_combinations_fail(self):
        for mode in ('bad','replace'):
            c=tiny(residual=True)
            if mode=='bad':c['model']['dual_quality_representation']['anchor_global_source']='typo'
            else:c['model']['quality_adapter']['mode']='replace'
            with self.assertRaises(ValueError):make(c)

    def test_quality_and_alignment_steps_obey_freeze_policy_and_freeze_buffers(self):
        for p in ('quality','alignment'):
            with self.subTest(phase=p):
                c=tiny(p);m=make(c);phases.configure(m,c);opt=build_optimizer(c,m)
                ids={id(p) for g in opt.param_groups for p in g['params']}
                self.assertEqual(ids,{id(p) for p in m.parameters() if p.requires_grad})
                m.train();phases.enforce_eval(m,c)
                self.assertTrue(m.iqa.shared_image_projector.net[-1].training)  # affine=False BN is ACTIVE
                pi_batches=m.iqa.shared_image_projector.net[-1].num_batches_tracked.clone()
                before=phases.frozen_digest(m);weights={n:v.detach().clone() for n,v in m.named_parameters()}
                with contextlib.redirect_stdout(io.StringIO()):
                    result=train_one_epoch(m,[batch()],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,1)
                self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
                self.assertEqual(before,phases.frozen_digest(m))
                self.assertGreater(m.iqa.shared_image_projector.net[-1].num_batches_tracked.item(),pi_batches.item())
                changed={n for n,v in m.named_parameters() if not torch.equal(weights[n],v)}
                self.assertTrue(changed)
                allowed=phases.QUALITY_PREFIXES+phases.QUALITY_RELATION_PREFIXES if p=='quality' else phases.ALIGN_PREFIXES
                self.assertTrue(all(n.startswith(allowed) for n in changed))
                if p=='alignment':
                    self.assertFalse(m.iqa.quality_adapter.training)
                    self.assertTrue(any(n.startswith('iqa.align_') for n in changed))
                else:self.assertTrue(any(n.startswith('iqa.quality_') for n in changed))

    def test_full_stage_transfer_restores_every_tensor_resets_optimizer_and_rejects_missing(self):
        c=tiny('quality');source=make(c)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'best.pth';torch.save({'model':source.state_dict(),'cfg':c,'run_id':c['run_id'],'epoch':3,'training_state':{'best_epoch':3}},p)
            a=tiny('alignment');a['train']['quality_stage_checkpoint']=str(p);a['output_dir']=tmp
            from utils.run_metadata import checkpoint_fingerprint
            manifest=Path(tmp)/'quality_stage_source.json'
            manifest.write_text(json.dumps({'full_sha256':checkpoint_fingerprint(p)['sha256'],'full_checkpoint':str(p),'selected_epoch':3}))
            target=make(a);report=phases.initialize(target,a)
            self.assertTrue(report['strict_full_model_transfer']);self.assertTrue(report['optimizer_reset'])
            for k,v in source.state_dict().items():torch.testing.assert_close(v,target.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(len(build_optimizer(a,target).state),0)
            phases.finish(target,report,tmp);self.assertTrue(json.loads((Path(tmp)/'phase_manifest.json').read_text())['frozen_state_unchanged'])
            broken=torch.load(p,weights_only=False);broken['model'].pop('iqa.nss_proj.0.weight');torch.save(broken,p)
            with self.assertRaisesRegex(ValueError,'fingerprint'):phases.initialize(make(a),a)
            manifest.write_text(json.dumps({'full_sha256':checkpoint_fingerprint(p)['sha256'],'full_checkpoint':str(p),'selected_epoch':3}))
            with self.assertRaises(RuntimeError):phases.initialize(make(a),a)

    def test_actual_trainer_main_runs_two_toy_phases_and_emits_usable_selected_checkpoints(self):
        import train_cm_pretrain as trainer
        from utils.run_metadata import checkpoint_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for phase in ('quality','alignment'):
                c=tiny(phase);directory=root/phase;c.update(output_dir=str(directory),results_csv=str(root/(phase+'.csv')),run_id=phase+'_toy')
                if phase=='alignment':
                    source=root/'quality'/'best_pretrain.pth';ck=torch.load(source,weights_only=False)
                    c['train'].update(quality_stage_checkpoint=str(source),quality_stage_run_id='quality_toy')
                    directory.mkdir()
                    (directory/'quality_stage_source.json').write_text(json.dumps({'full_checkpoint':str(source),'full_sha256':checkpoint_fingerprint(source)['sha256'],'selected_epoch':ck['epoch']}))
                cp=root/(phase+'.yaml');cp.write_text(yaml.safe_dump(c))
                # A two-step synthetic CPU fixture exercises the actual main,
                # optimizer, phase loader, selector and checkpoint writer.
                m=make(c)
                with patch.object(sys,'argv',['train_cm_pretrain.py','--config',str(cp),'--fixed-epochs','2']),patch.object(trainer,'build_dataloaders',return_value=([batch()],[batch()],[batch()])),patch('models.two_stage_models.GlobalVisualEncoder',Global),patch('models.two_stage_models.ResNet50LocalEncoder',Local),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text),patch('torch.cuda.is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
                report=json.loads((directory/'phase_manifest.json').read_text())
                self.assertTrue(report['frozen_state_unchanged'])
                self.assertTrue((directory/('best_component_anchor_pretrain.pth' if phase=='quality' else 'best_align_pretrain.pth')).is_file())

    def test_phase_validation_rejects_wrong_losses_selection_and_source(self):
        for p,key,value in [('quality','lambda_img_prompt',.2),('alignment','lambda_artifact_anchor',.1),('alignment','pretrain_selection_metric','val_loss'),('alignment','quality_stage_checkpoint',None)]:
            c=tiny(p);c['train'][key]=value
            with self.assertRaises(ValueError):phases.configure(make(c),c)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.ComponentStrengthTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root
        shutil.copytree(sweep.ROOT/'configs/quality_oas_controls',self.root/'configs/quality_oas_controls')
        p=patch.object(sweep,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def fake_pretrain(self,r):
        case=self
        class Process:
            def __init__(self,command,**kwargs):
                c=r.validate('pretrain');d,t=r.paths('pretrain');phase=c['train'].get('pretraining_phase','joint')
                metric=c['train']['pretrain_selection_metric'];field='align' if phase=='alignment' else 'component_anchor'
                filename='best_align_pretrain.pth' if phase=='alignment' else 'best_component_anchor_pretrain.pth'
                (d/'config_snapshot.yaml').write_text(yaml.safe_dump(c))
                state={prefix+'weight':torch.zeros(1) for prefix in ('iqa.backbone.global_encoder.','iqa.backbone.local_encoder.','iqa.backbone.fusion.semantic_proj.','iqa.quality_adapter.','iqa.quality_local_encoder.','iqa.nss_proj.','iqa.nss_gate.','iqa.shared_image_projector.','iqa.shared_text_projector.')}
                torch.save(dict(cfg=c,run_id=c['run_id'],epoch=1,checkpoint_scope='representation',selection_metric=metric,model=state),d/filename)
                if phase=='quality':torch.save({'model':state,'epoch':1},d/'best_pretrain.pth')
                if phase!='joint':
                    source=None if phase=='quality' else {'sha256':r.read_json(d/'quality_stage_source.json')['full_sha256']}
                    r.write_new(d/'phase_manifest.json',dict(phase=phase,source=source,strict_full_model_transfer=phase=='alignment',frozen_state_unchanged=True,frozen_sha256_before='same',frozen_sha256_after='same'))
                with t.open('w') as f:
                    w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                    for epoch,val in [(1,.7),(2,.6)]:w.writerow(dict(event='epoch',epoch=epoch,metrics_json=json.dumps({metric:val})))
                    w.writerow(dict(event='final',epoch=2,metrics_json=json.dumps(dict(selection_metric=metric,best_epoch=1,last_epoch=2,**{'best_'+field+'_epoch':1}))))
                self.stdout=iter(['fake pretrain complete\n'])
            def wait(self):return 0
        return Process

    def pretrain(self,v):
        r=sweep.make_runner(v)
        with patch('torch.cuda.is_available',return_value=True),patch.object(sweep.subprocess,'Popen',self.fake_pretrain(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        return r

    def test_configs_independent_and_downstream_preserves_baseline(self):
        for v in sweep.VARIANTS:
            r=sweep.make_runner(v)
            for s in sweep.STAGES:r.validate(s)
            for s in sweep.STAGES[1:]:
                c=r.validate(s);ref=yaml.safe_load((self.root/f'configs/current_tuning_reference/{s}.yaml').read_text())
                for k in ('run_id','output_dir','results_csv'):ref[k]=c[k]
                ref['train']['pretrained_checkpoint']=c['train']['pretrained_checkpoint'];self.assertEqual(c,ref)
                self.assertFalse(c['model']['global_encoder']['freeze']);self.assertFalse(c['model']['local_encoder']['freeze'])
        self.assertEqual(sweep.expected_config('quality_first')['train']['pretrain_selection_metric'],'val_align_proxy_SROCC')
        self.assertEqual(sweep.expected_config('residual_oas')['model']['dual_quality_representation']['anchor_global_source'],'residual')

    def test_dry_plan_orders_stages_cpu_refuses_and_preserves_partial_outputs(self):
        with patch.object(sweep.subprocess,'Popen',side_effect=AssertionError('no training')),contextlib.redirect_stdout(io.StringIO()) as out:sweep.run_pipeline('quality_first','all',True)
        plan=json.loads(out.getvalue());self.assertEqual(len(plan['pending']),5)
        self.assertIn('quality_stage',plan['pending'][0][0]);self.assertIn('quality_first',plan['pending'][1][0])
        for v in ('quality_stage','residual_oas'):
            with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):sweep.make_runner(v).run('pretrain')
        d,_=sweep.make_runner('quality_first').paths('agiqa3k');d.mkdir(parents=True)
        with patch.object(sweep.subprocess,'Popen',side_effect=AssertionError('no training')),self.assertRaises(FileNotFoundError):sweep.run_pipeline('quality_first','all')
        self.assertTrue(d.exists())

    def test_fake_quality_to_alignment_source_pack_and_tampering_rejection(self):
        quality=self.pretrain('quality_stage');final=self.pretrain('quality_first');source=final.completed('pretrain')
        self.assertEqual(source['quality_stage'],quality.completed('pretrain'))
        with contextlib.redirect_stdout(io.StringIO()):final.pack('pretrain')
        archive=self.root/f'{final.VERSION}_pretrain_results.tar.gz'
        with tarfile.open(archive) as tar:
            names=tar.getnames();self.assertEqual(sum(x.endswith('.csv') for x in names),2)
            self.assertTrue(any(x.endswith('phase_manifest.json') for x in names));self.assertFalse(any(x.endswith('.pth') for x in names))
        d,_=quality.paths('pretrain');(d/'best_pretrain.pth').write_bytes(b'changed')
        with self.assertRaises(ValueError):final.completed('pretrain')

    def test_residual_fake_downstream_pipeline_source_and_pack(self):
        r=self.pretrain('residual_oas');source=r.completed('pretrain')
        for stage in sweep.STAGES[1:]:
            class Downstream:
                def __init__(inner,command,**kwargs):
                    c=r.validate(stage);d,t=r.paths(stage);(d/'config_snapshot.yaml').write_text(yaml.safe_dump(c))
                    r.write_new(d/'run_metadata.json',dict(seed=42,split_seed=c['split_seed'],checkpoint_verified_after_load=True,checkpoint={'sha256':source['sha256']}))
                    from utils.agiqa1k_subset1078 import expected_resplit_manifest
                    if c['train'].get('record_split_manifest'):
                        r.write_new(d/'data_split_manifest.json',{'train_seed':42,'split_seed':c['split_seed'],'indices':expected_resplit_manifest(3407)['split_indices'] if stage=='agiqa1k' else {}})
                    with t.open('w') as f:
                        w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                        for e in range(1,61):w.writerow(dict(event='epoch',epoch=e,metrics_json='{}'))
                        w.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
                    inner.stdout=iter(['fake downstream\n'])
                def wait(inner):return 0
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Downstream),patch.object(r.subprocess,'run'),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run(stage)
            self.assertEqual(r.completed(stage),source)
        with contextlib.redirect_stdout(io.StringIO()):r.pack('downstream')
        with tarfile.open(self.root/f'{r.VERSION}_downstream_results.tar.gz') as tar:self.assertEqual(sum(x.endswith('.csv') for x in tar.getnames()),4)

if __name__=='__main__':unittest.main()
