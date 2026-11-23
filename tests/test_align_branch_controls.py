"""Real small-model branch routing, gradients, transfer and protocol checks."""
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
from scripts import run_align_branch_controls as study
from test_projection_only import config as tiny
from test_frozen_clip_baseline import small_model
from test_downstream_pipeline_controls import batch
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import compute_losses,build_optimizer,evaluate
from utils.quality_representation import dual_quality_diagnostics
import test_align_tuning as fixture


def configuration(variant=None):
    r=study.make_runner();real=r.config(variant or 'aglobal','agiqa3k');c=tiny(True)
    c['train']=copy.deepcopy(real['train']);c['model']['align_adapter']=copy.deepcopy(real['model']['align_adapter'])
    if variant:c['model']['align_branch_control']=copy.deepcopy(real['model']['align_branch_control'])
    return c


class ModelTests(unittest.TestCase):
    def setUp(self):
        saved=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,saved);torch.manual_seed(42)

    def test_same_parameters_initialization_rng_and_quality_eval_outputs(self):
        torch.manual_seed(42);base=small_model(configuration()).eval();rng=torch.random.get_rng_state()
        image=torch.randn(4,3,16,16);prompts=['cat','dog','bird','fish'];original=base.forward_single(image,prompts)
        for v in study.VARIANTS:
            torch.manual_seed(42);m=small_model(configuration(v)).eval()
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(m.state_dict().keys(),base.state_dict().keys())
            for k,value in m.state_dict().items():torch.testing.assert_close(value,base.state_dict()[k],rtol=0,atol=0)
            out=m.forward_single(image,prompts)
            for k in ('s_quality','s_quality_global','s_quality_patch','quality_representation','s_align_global'):
                torch.testing.assert_close(out[k],original[k],rtol=0,atol=0)
            if v!='auniform':torch.testing.assert_close(out['s_align_patch'],original['s_align_patch'],rtol=0,atol=0)

    def test_global_half_skip_gate_and_keep_real_local_auxiliary_gradients(self):
        data=batch();data['image']=torch.randn(4,3,16,16)
        for v,weight in [('aglobal',1.),('ahalf',.5)]:
            c=configuration(v);m=small_model(c).eval()
            with patch.object(m.align_gate,'forward',side_effect=AssertionError('gate executed')):
                out,loss=compute_losses(m,data,'cpu',c)
            torch.testing.assert_close(out['s_align'],weight*out['s_align_global']+(1-weight)*out['s_align_patch'])
            self.assertTrue(torch.all(out['align_gate']==weight));self.assertIsNone(out['align_gate_logit'])
            p=m.align_patch_head[0].weight
            main=torch.autograd.grad(out['s_align'].sum(),p,retain_graph=True,allow_unused=True)[0]
            if v=='aglobal':self.assertIsNone(main)
            full=torch.autograd.grad(loss['loss_align_task'],p,retain_graph=True)[0]
            self.assertGreater(full.abs().sum(),0) # lambda_align_aux is retained
            opt=build_optimizer(c,m);ids={id(p) for g in opt.param_groups for p in g['params']}
            self.assertFalse(any(id(p) in ids for p in m.align_gate.parameters()))
            m.train();self.assertFalse(m.align_gate.training)

    def test_uniform_skips_matching_projections_and_means_unprojected_local_tokens(self):
        c=configuration('auniform');m=small_model(c).eval();tokens=[];pairs=[]
        h=m.align_patch_adapter.register_forward_hook(lambda mod,ins,out:tokens.append(out))
        h2=m.align_patch_head.register_forward_pre_hook(lambda mod,ins:pairs.append(ins[0]))
        with patch.object(m.align_patch_img_proj,'forward',side_effect=AssertionError('image matching executed')),patch.object(m.align_patch_txt_proj,'forward',side_effect=AssertionError('text matching executed')):
            out=m.forward_single(torch.randn(4,3,16,16),['cat','dog','bird','fish'])
        h.remove();h2.remove();n=out['align_patch_weights'].shape[1]
        torch.testing.assert_close(out['align_patch_weights'],torch.full_like(out['align_patch_weights'],1/n))
        torch.testing.assert_close(pairs[0][:,:tokens[0].shape[-1]],tokens[0].mean(1))
        self.assertIsNone(out['align_patch_features']);self.assertIsNone(out['align_text_feature'])
        self.assertTrue(all(not p.requires_grad for mod in (m.align_patch_img_proj,m.align_patch_txt_proj) for p in mod.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.align_gate.parameters()))

    def test_strict_baseline_representation_transfer_and_real_evaluation_diagnostics(self):
        base=small_model(configuration()).eval();state={'iqa.'+k:v for k,v in base.state_dict().items()};state=_extract_representation_state(state)
        for v in study.VARIANTS:
            c=configuration(v);m=small_model(c).eval()
            with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(m,'baseline.pth',load_scope='representation')
            for k,val in state.items():torch.testing.assert_close(m.state_dict()[k.removeprefix('iqa.')],val,rtol=0,atol=0)
            data=batch();data['image']=torch.randn(4,3,16,16)
            metrics=evaluate(m,[data],'cpu',c)
            flags=dual_quality_diagnostics(m)
            for k,val in flags.items():self.assertEqual(metrics['val_'+k],val)
            self.assertEqual(flags['align_control_gate_trainable'],float(v=='auniform'))
            self.assertEqual(flags['align_control_match_trainable'],float(v!='auniform'))

    def test_invalid_combinations_fail_explicitly(self):
        for kind in ('unknown','version','combined','pretrain','gate_loss','quality_only'):
            c=configuration('aglobal')
            if kind=='unknown':c['model']['align_branch_control']['oops']=True
            elif kind=='version':c['model']['align_branch_control']['version']='old'
            elif kind=='combined':c['model']['align_branch_control']['local_pool']='uniform'
            elif kind=='pretrain':c['train']['use_joint_iqa_pretraining']=True
            elif kind=='gate_loss':c['train']['lambda_dynamic_align_gate']=.1
            else:c['train']['quality_only']=True
            with self.assertRaises(ValueError):small_model(c)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.r=study.make_runner();p=patch.object(fixture,'r',self.r);p.start();self.addCleanup(p.stop)
        self.f=fixture.RunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root;self.source=self.f.source
        shutil.copy(study.ROOT/'docs/align_branch_controls.md',self.root/'docs/align_branch_controls.md')

    def artifact(self,v,ds):
        r=self.r;d,t=self.f.fake(v,ds,False);control=r.config(v,ds)['model']['align_branch_control']
        flags={'global_output':float(control['output']=='global'),'half_mix':float(control['output']=='half'),
            'uniform_pool':float(control['local_pool']=='uniform'),'gate_trainable':float(control['output']=='learned'),
            'match_trainable':float(control['local_pool']=='text')}
        gate=1. if v=='aglobal' else .5 if v=='ahalf' else .9
        with t.open() as f:rows=list(csv.DictReader(f))
        bt=None
        for row in rows:
            if row['event'] not in ('epoch','best_val','best_test'):continue
            m=json.loads(row['metrics_json']);pre='val_val' if row['event']=='epoch' else 'val' if row['event']=='best_val' else 'test'
            m.update({pre+'_align_control_'+k:x for k,x in flags.items()});m[pre+'_align_gate']=gate;row['metrics_json']=json.dumps(m)
            if row['event']=='best_test':bt=m
        final=next(x for x in rows if x['event']=='final');m=json.loads(final['metrics_json']);m['best_test']=bt;final['metrics_json']=json.dumps(m)
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.seal(d,t);return d,t

    def test_only_one_model_factor_and_all_losses_lrs_keep_baseline(self):
        r=self.r
        for ds in r.DATASETS:
            base=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            for v in r.VARIANTS:
                c=r.config(v,ds);expected=copy.deepcopy(base)
                for k in ('run_id','output_dir','results_csv'):expected[k]=c[k]
                expected['model']['align_branch_control']=r.VARIANTS[v]
                expected['train'].update(record_split_manifest=True,record_optimizer_initialization=True)
                self.assertEqual(c,expected)
        with patch.object(r,'control'),patch.object(r,'run') as launch:r.run_all('all')
        self.assertEqual(launch.call_count,6)
        with patch.object(r,'source',side_effect=AssertionError('no weights')),contextlib.redirect_stdout(io.StringIO()):r.run_all('all',True)

    def test_six_fake_completions_pack_and_wrong_actual_flags_refused(self):
        r=self.r;controls={}
        for ds in r.DATASETS:
            for v in r.VARIANTS:
                d,t=self.artifact(v,ds);r.verify(v,ds,None,self.source);controls.setdefault(ds,(d,t))
        with patch.object(r,'control',side_effect=lambda ds,s,fp:controls[ds]),contextlib.redirect_stdout(io.StringIO()):r.pack()
        with tarfile.open(self.root/f'{r.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),6)
            self.assertFalse(any(n.endswith('.pth') for n in names))
        d,t=r.paths('aglobal','agiqa3k')
        with t.open() as f:rows=list(csv.DictReader(f))
        row=next(x for x in rows if x['event']=='epoch');m=json.loads(row['metrics_json']);m['val_val_align_control_gate_trainable']=1.;row['metrics_json']=json.dumps(m)
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'control'):r.verify('aglobal','agiqa3k',None,self.source)

if __name__=='__main__':unittest.main()
