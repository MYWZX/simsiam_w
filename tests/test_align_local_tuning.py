"""Independent auxiliary/temperature/context controls; no real data training."""
import contextlib
import copy
import csv
import io
import json
import shutil
import tarfile
import unittest
from unittest.mock import patch

import torch
import yaml
from scripts import run_align_local_tuning as study
from utils.alignment_controls import alignment_aux_weights
from test_projection_only import config as tiny
from test_frozen_clip_baseline import small_model
from test_downstream_pipeline_controls import ToyScores,batch
from train_cm_iqa import compute_losses,build_optimizer
from utils.quality_representation import dual_quality_diagnostics
import test_align_tuning as fixture


class ModelTests(unittest.TestCase):
    def setUp(self):
        state=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,state);torch.manual_seed(42)

    def test_legacy_auxiliary_defaults_and_independent_local_loss_gradient(self):
        m=ToyScores();data=batch();c={'train':{'two_output_only':True,'lambda_quality':1.,'lambda_align':1.,'lambda_align_aux':.3,'lambda_neg':0.}}
        _,old=compute_losses(m,data,'cpu',c)
        same=copy.deepcopy(c);same['train'].update(lambda_align_global_aux=.3,lambda_align_local_aux=.3)
        _,equal=compute_losses(m,data,'cpu',same)
        torch.testing.assert_close(old['loss'],equal['loss'],rtol=0,atol=0)
        for coefficient in (.15,.6):
            new=copy.deepcopy(same);new['train']['lambda_align_local_aux']=coefficient
            _,loss=compute_losses(m,data,'cpu',new)
            torch.testing.assert_close(loss['loss']-old['loss'],(coefficient-.3)*old['align_patch'],atol=1e-7,rtol=1e-5)
            torch.testing.assert_close(loss['loss_quality_task'],old['loss_quality_task'],rtol=0,atol=0)
            torch.testing.assert_close(loss['align_global'],old['align_global'],rtol=0,atol=0)
            torch.testing.assert_close(loss['weighted_align_patch'],coefficient*loss['align_patch'])
            self.assertTrue(torch.isfinite(torch.autograd.grad(loss['loss'],m.weight)[0]).all())
        self.assertEqual(alignment_aux_weights({'lambda_align_aux':.7}),(.7,.7))
        for value in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):alignment_aux_weights({'lambda_align_local_aux':value})

    def test_actual_temperature_formula_and_default_parameter_rng_equivalence(self):
        c=tiny(True);c['train']=copy.deepcopy(study.make_runner().config('amatchtau050','agiqa3k')['train'])
        c['model']['align_adapter']=copy.deepcopy(study.make_runner().config('amatchtau050','agiqa3k')['model']['align_adapter'])
        torch.manual_seed(42);base=small_model(c).eval();rng=torch.random.get_rng_state()
        image=torch.randn(4,3,16,16);prompts=['cat','dog','bird','fish'];old=base.forward_single(image,prompts)
        for tau in (1.,.5,2.):
            changed=copy.deepcopy(c);changed['model']['align_local_temperature']=tau
            torch.manual_seed(42);m=small_model(changed).eval();self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            for k,v in base.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
            out=m.forward_single(image,prompts)
            logits=(out['align_patch_features']*out['align_text_feature'].unsqueeze(1)).sum(-1)
            torch.testing.assert_close(out['align_patch_weights'],torch.softmax(logits/tau,dim=1))
            for k in ('s_quality','s_quality_global','s_quality_patch','s_align_global'):
                torch.testing.assert_close(old[k],out[k],rtol=0,atol=0)
            if tau==1.:torch.testing.assert_close(old['s_align'],out['s_align'],rtol=0,atol=0)
            else:
                self.assertFalse(torch.equal(old['align_patch_weights'],out['align_patch_weights']))
                self.assertEqual(dual_quality_diagnostics(m)['align_local_temperature'],tau)
        for tau in (0,-1,float('nan'),float('inf')):
            changed=copy.deepcopy(c);changed['model']['align_local_temperature']=tau
            with self.assertRaises(ValueError):small_model(changed)
        changed=copy.deepcopy(c);changed['model']['align_local_temperature']=.5;changed['train']['use_joint_iqa_pretraining']=True
        with self.assertRaises(ValueError):small_model(changed)

    def test_actual_context_group_lr_only_changes_and_all_memberships_match(self):
        r=study.make_runner();c=tiny(True);c['train']=copy.deepcopy(r.config('amatchtau050','agiqa3k')['train']);m=small_model(c)
        base=build_optimizer(c,m);members={g['name']:{id(p) for p in g['params']} for g in base.param_groups}
        for v in ('actxlr050','actxlr200'):
            cc=copy.deepcopy(c);cc['train']=r.config(v,'agiqa3k')['train'];opt=build_optimizer(cc,m)
            for g in opt.param_groups:
                self.assertEqual({id(p) for p in g['params']},members[g['name']]);self.assertAlmostEqual(g['lr'],r.expected_group_lrs(cc)[g['name']])
            owner={id(p):g for g in opt.param_groups for p in g['params']}
            self.assertAlmostEqual(owner[id(m.backbone.text_encoder.context_vectors)]['lr'],5e-6 if v=='actxlr050' else 2e-5)
            self.assertEqual(owner[id(m.shared_image_projector.net[0].weight)]['lr'],1e-4)
            self.assertEqual(owner[id(m.backbone.fusion.semantic_proj.weight)]['lr'],1e-5)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.r=study.make_runner();p=patch.object(fixture,'r',self.r);p.start();self.addCleanup(p.stop)
        self.f=fixture.RunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root;self.source=self.f.source
        shutil.copy(study.ROOT/'docs/align_local_tuning.md',self.root/'docs/align_local_tuning.md')

    def artifact(self,v,ds):
        r=self.r;d,t=self.f.fake(v,ds,False);c=r.config(v,ds)
        global_weight,local_weight=alignment_aux_weights(c['train'])
        with t.open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            if row['event']!='epoch':continue
            m=json.loads(row['metrics_json'])
            m['train_train_loss']+=(global_weight-.3)*m['train_train_align_global']+(local_weight-.3)*m['train_train_align_patch']
            m['train_train_weighted_align_patch']=local_weight*m['train_train_align_patch']
            if study.VARIANTS[v][0]=='temperature':m['val_val_align_local_temperature']=study.VARIANTS[v][1]
            row['metrics_json']=json.dumps(m)
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.seal(d,t);return d,t

    def test_all_twelve_configs_single_factor_and_original_source_restored(self):
        r=self.r;paths=[]
        for ds in r.DATASETS:
            baseline=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            for v,(kind,value) in study.VARIANTS.items():
                actual=r.config(v,ds);expected=copy.deepcopy(baseline)
                for k in ('run_id','output_dir','results_csv'):expected[k]=actual[k]
                expected['train'].update(record_split_manifest=True,record_optimizer_initialization=True)
                if kind=='aux':expected['train'].update(lambda_align_global_aux=.3,lambda_align_local_aux=value)
                elif kind=='temperature':expected['model']['align_local_temperature']=value
                else:expected['train']['layerwise_lr']['text_context_mult']=value
                self.assertEqual(actual,expected);paths.append(actual['run_id'])
        self.assertEqual(len(set(paths)),12)
        for queue,count in [('all',12),('low',6),('high',6),('aux',4),('match',4),('context',4)]:
            with patch.object(r,'control'),patch.object(r,'run') as launch:r.run_all(queue)
            self.assertEqual(launch.call_count,count)

    def test_twelve_fake_completions_pack_and_effective_local_weight_refusal(self):
        r=self.r;controls={}
        for ds in r.DATASETS:
            for v in r.VARIANTS:
                d,t=self.artifact(v,ds);r.verify(v,ds,None,self.source);controls.setdefault(ds,(d,t))
        with patch.object(r,'control',side_effect=lambda ds,s,fp:controls[ds]),contextlib.redirect_stdout(io.StringIO()):r.pack()
        with tarfile.open(self.root/f'{r.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),12);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,t=r.paths('alocalaux050','agiqa3k')
        with t.open() as f:rows=list(csv.DictReader(f))
        row=next(x for x in rows if x['event']=='epoch');m=json.loads(row['metrics_json']);m['train_train_weighted_align_patch']*=2;row['metrics_json']=json.dumps(m)
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'local alignment'):r.verify('alocalaux050','agiqa3k',None,self.source)

    def test_cpu_refusal_dry_plan_and_wrong_temperature_refusal(self):
        r=self.r
        with patch.object(r,'source',side_effect=AssertionError('no weights')),patch.object(r,'run',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):r.run_all('all',True)
        with patch.object(r,'control'),patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('actxlr050','agiqa3k')
        d,t=self.artifact('amatchtau050','agiqa3k')
        with t.open() as f:rows=list(csv.DictReader(f))
        row=next(x for x in rows if x['event']=='epoch');m=json.loads(row['metrics_json']);m['val_val_align_local_temperature']=2;row['metrics_json']=json.dumps(m)
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'temperature'):r.verify('amatchtau050','agiqa3k',None,self.source)

if __name__=='__main__':unittest.main()
