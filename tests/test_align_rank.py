"""Independent rank-weight protocol and same-epoch branch diagnostics."""
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
from scripts import run_align_rank as study
from scripts import run_align_tuning as previous
from scripts.audit_align_branches import audit_csv
from test_downstream_pipeline_controls import ToyScores,batch
from train_cm_iqa import compute_losses
import test_align_tuning as fixture


class ModelAndConfigTests(unittest.TestCase):
    def test_single_factor_config_and_independent_protocols(self):
        r=study.make_runner();other=study.make_runner()
        self.assertIsNot(r,other);self.assertEqual(len(previous.VARIANTS),4)
        self.assertEqual(previous.VERSION,'projection_only_v1_v2_align_tuning')
        names=[]
        for ds in r.DATASETS:
            original=yaml.safe_load((r.ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            for v,weight in study.VARIANTS.items():
                c=r.config(v,ds);expected=copy.deepcopy(original)
                for k in ('output_dir','results_csv','run_id'):expected[k]=c[k]
                expected['train'].update(lambda_rank_align=weight,record_split_manifest=True,record_optimizer_initialization=True)
                self.assertEqual(c,expected);names.append(c['run_id'])
                self.assertEqual(c['train']['layerwise_lr']['align_mult'],1.)
                self.assertEqual(c['train']['lambda_srocc_align'],.04)
                self.assertEqual(r.expected_group_lrs(c)['align'],1e-4)
                self.assertEqual(r.expected_group_lrs(c)['shared_projection'],1e-4)
        self.assertEqual(len(set(names)),4)
        with self.assertRaises(ValueError):r.config('arank050','agiqa1k')

    def test_real_pairwise_loss_weight_changes_only_alignment_objective(self):
        saved=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,saved);torch.manual_seed(42)
        r=study.make_runner();model=ToyScores();data=batch()
        data['image']=torch.tensor([-.8,-.1,.15,.5])[:,None,None,None].expand(-1,3,8,8).clone()
        data['align_score']=torch.tensor([.15,.65,.4,.85])
        c=r.config('arank050','agiqa3k');c['train']['lambda_rank_align']=.03
        output,base=compute_losses(model,data,'cpu',c)
        basegrad=torch.autograd.grad(base['loss'],model.weight,retain_graph=True)[0]
        self.assertGreater(base['rank_align'].item(),0)
        for v,coef in study.VARIANTS.items():
            out,loss=compute_losses(model,data,'cpu',r.config(v,'agiqa3k'))
            torch.testing.assert_close(loss['loss']-base['loss'],(coef-.03)*base['rank_align'],atol=1e-7,rtol=1e-5)
            torch.testing.assert_close(loss['loss_quality_task'],base['loss_quality_task'],rtol=0,atol=0)
            for k in ('quality','align','rank_align','srocc_quality','srocc_align'):
                torch.testing.assert_close(loss[k],base[k],rtol=0,atol=0)
            for k in output:
                if isinstance(output[k],torch.Tensor):torch.testing.assert_close(output[k],out[k],rtol=0,atol=0)
            gradient=torch.autograd.grad(loss['loss'],model.weight)[0]
            self.assertTrue(torch.isfinite(gradient).all());self.assertFalse(torch.equal(gradient,basegrad))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.r=study.make_runner()
        p=patch.object(fixture,'r',self.r);p.start();self.addCleanup(p.stop)
        self.f=fixture.RunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root;self.source=self.f.source
        shutil.copy(study.ROOT/'docs/align_rank.md',self.root/'docs/align_rank.md')

    def test_four_fake_runs_pack_and_old_results_not_required(self):
        r=self.r;controls={}
        for ds in r.DATASETS:
            for v in r.VARIANTS:
                d,t=self.f.fake(v,ds);r.verify(v,ds,None,self.source);controls.setdefault(ds,(d,t))
        with patch.object(r,'control',side_effect=lambda ds,s,fp:controls[ds]),contextlib.redirect_stdout(io.StringIO()):r.pack()
        with tarfile.open(self.root/f'{r.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertEqual(sum(x.endswith('.csv') for x in names),4)
            self.assertIn('docs/align_rank.md',names);self.assertNotIn('docs/align_tuning.md',names)
            self.assertFalse(any(x.endswith('.pth') for x in names))
        self.assertFalse((self.root/'projection_only_v1_v2_align_tuning_results.tar.gz').exists())

    def test_queue_plans_cpu_guard_and_partial_output_refusal(self):
        r=self.r
        for queue,count in [('all',4),('low',2),('high',2)]:
            with patch.object(r,'control'),patch.object(r,'run') as launch:r.run_all(queue)
            self.assertEqual(launch.call_count,count)
        with patch.object(r,'source',side_effect=AssertionError('no source load')),patch.object(r,'run',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):r.run_all('all',True)
        with patch.object(r,'control'),patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('arank050','agiqa3k')
        d,_=r.paths('arank200','aigciqa2023');d.mkdir(parents=True)
        with patch.object(r,'control'),patch.object(r,'run') as launch:
            with self.assertRaises(FileExistsError):r.run_all('all')
            launch.assert_not_called()
            with self.assertRaises(FileNotFoundError):r.run_all('all',continue_completed=True)
            launch.assert_not_called()
        self.assertTrue(d.exists())

    def test_reject_carryover_lr_srocc_and_effective_rank_tampering(self):
        r=self.r;p=self.root/'configs/current_tuning_reference/agiqa3k.yaml';original=yaml.safe_load(p.read_text())
        for kind in ('lr','srocc'):
            c=copy.deepcopy(original)
            if kind=='lr':c['train']['layerwise_lr']['align_mult']=2.
            else:c['train']['lambda_srocc_align']=.08
            p.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):r.config('arank050','agiqa3k')
        p.write_text(yaml.safe_dump(original));d,t=self.f.fake('arank050','agiqa3k')
        with t.open() as f:rows=list(csv.DictReader(f))
        row=next(x for x in rows if x['event']=='epoch');m=json.loads(row['metrics_json']);m['train_train_loss']+=.1;row['metrics_json']=json.dumps(m)
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'objective'):r.verify('arank050','agiqa3k',None,self.source)

    def test_branch_audit_same_epoch_finite_metrics_and_global_gate_mean(self):
        _,t=self.f.fake('arank050','agiqa3k')
        with t.open() as f:rows=list(csv.DictReader(f))
        target=None
        for row in rows:
            if row['event'] not in ('best_val','best_test'):continue
            prefix='val' if row['event']=='best_val' else 'test'
            data={f'{prefix}_align{suffix}_{metric}':value for suffix,value in [('',.8),('_global',.81),('_patch',.65)] for metric in ('SROCC','PLCC')}
            data[f'{prefix}_align_gate']=.93;row['metrics_json']=json.dumps(data)
            if prefix=='test':target=data
        final=next(x for x in rows if x['event']=='final');data=json.loads(final['metrics_json']);data['best_test']=target;final['metrics_json']=json.dumps(data)
        def write():
            with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        write();report=audit_csv(t)
        self.assertEqual(report['selected_epoch'],40);self.assertAlmostEqual(report['test']['mean_local_gate'],.07)
        self.assertAlmostEqual(report['test']['mixed_minus_global_srocc'],-.01)
        row=next(x for x in rows if x['event']=='best_test');row['epoch']='30';write()
        with self.assertRaises(ValueError):audit_csv(t)

if __name__=='__main__':unittest.main()
