"""Quality Soft-SROCC weight changes with fixed source/splits; no real training."""
import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
import test_projection_only as projection
from test_projection_ablations import make
from train_cm_iqa import build_optimizer
from utils.run_metadata import optimizer_initialization
from utils.refit_protocol import digest
from scripts import run_quality_srocc as runner

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        shutil.copytree(runner.ROOT/'configs/projection_only_v1_baseline',self.root/'configs/projection_only_v1_baseline')
        (self.root/'docs').mkdir();(self.root/'docs/current_baseline.json').write_text('{}');(self.root/'docs/quality_srocc.md').write_text('protocol')
        p=patch.object(runner,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.source={'sha256':'pinned-source'}
        p=patch.object(runner,'source',return_value=self.source);p.start();self.addCleanup(p.stop)

    def fake(self,v,ds,split):
        c=runner.config(v,ds,split);d,table=runner.paths(v,ds,split);d.mkdir(parents=True)
        (d/'launch_config.yaml').write_text(yaml.safe_dump(c));(d/'config_snapshot.yaml').write_text(yaml.safe_dump(c));(d/'console.log').write_text('fake log')
        runner.base.write_new(d/'source_fingerprint.json',self.source)
        runner.base.write_new(d/'launch_manifest.json',dict(version=runner.VERSION,variant=v,dataset=ds,split_seed=c['split_seed'],source=self.source,config_sha256=runner.base.digest(d/'launch_config.yaml')))
        sizes=dict(train=755,val=108,test=215) if ds=='agiqa1k' else dict(train=7,val=1,test=2)
        order=torch.randperm(sum(sizes.values()),generator=torch.Generator().manual_seed(c['split_seed'])).tolist();n,va=sizes['train'],sizes['val']
        indices=dict(train=order[:n],val=order[n:n+va],test=order[n+va:])
        evidence=dict(train_seed=42,split_seed=c['split_seed'],indices=indices,split_sizes=sizes,split_indices_sha256=digest(indices),dataset_ids_sha256='same-order')
        runner.base.write_new(d/'data_split_manifest.json',evidence)
        record=[dict(name=name,lr=lr,weight_decay=.002,parameter_count=1) for name,lr in runner.expected_group_lrs(c).items()]
        runner.base.write_new(d/'run_metadata.json',dict(seed=42,split_seed=c['split_seed'],checkpoint_verified_after_load=True,checkpoint=self.source,
            optimizer_initialization=record,data_split={k:v for k,v in evidence.items() if k!='indices'}))
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader()
            writer.writerow(dict(event='optimizer_initialization',metrics_json=json.dumps({'groups':record})))
            for epoch in range(1,61):
                metrics=dict(train_train_srocc_quality=1.2,train_train_weighted_quality_srocc=1.2*c['train']['lambda_srocc_quality'],
                    train_train_rank_quality=.5,train_train_weighted_quality_rank=.5*c['train']['lambda_rank_quality'],
                    train_train_quality=.02,train_train_weighted_quality_mse=.02*c['train']['lambda_quality'])
                writer.writerow(dict(event='epoch',epoch=epoch,metrics_json=json.dumps(metrics)))
            writer.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in runner.FILES if n!='completion_manifest.json'},csv_sha256=runner.base.digest(table)))
        (d/'unused_weights.pth').write_bytes(b'excluded')
        return d,table

    def test_fixed_splits_config_and_cpu_refusal(self):
        with self.assertRaises(ValueError):runner.config('qsrocc050','agiqa1k',3047)
        with self.assertRaises(ValueError):runner.config('qsrocc050','agiqa3k',3407)
        for ds in runner.DATASETS:
            original=yaml.safe_load((self.root/f'configs/projection_only_v1_baseline/{ds}.yaml').read_text())
            for v in runner.VARIANTS:
                c=runner.config(v,ds)
                self.assertEqual(c['seed'],42);self.assertEqual(c['split_seed'],3407 if ds=='agiqa1k' else 42)
                self.assertEqual(c['model'],original['model'])
                self.assertEqual(c['train']['lambda_srocc_quality'],original['train']['lambda_srocc_quality']*runner.VARIANTS[v])
                self.assertEqual(c['train']['lambda_rank_quality'],original['train']['lambda_rank_quality'])
                self.assertEqual(c['train']['srocc_tau'],original['train']['srocc_tau'])
                self.assertEqual(c['train']['weight_decay'],.002)
                self.assertNotIn('post_load_qa_alpha',c['train'])
                expected=dict(original['train']['layerwise_lr']);expected=dict(original['train']['layerwise_lr'])
                self.assertEqual(c['train']['layerwise_lr'],expected)
        with patch.object(runner,'control',side_effect=ValueError('missing control')):
            with self.assertRaisesRegex(ValueError,'missing control'):runner.run('qsrocc050','agiqa1k',dry_run=True)
        with patch.object(runner,'control'),contextlib.redirect_stdout(io.StringIO()):runner.run('qsrocc050','agiqa3k',dry_run=True)
        with patch.object(runner,'control'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):runner.run('qsrocc050','agiqa3k')

    def test_control_provenance_pack_and_no_overwrite(self):
        controls={}
        for ds in runner.DATASETS:controls[ds]=self.fake('qsrocc050',ds,3407 if ds=='agiqa1k' else None)
        # Replace their control labels only in the mocked control resolver; reuse
        # identical actual memberships to exercise package checks.
        for ds in runner.DATASETS:
            for v in ('qsrocc050','qsrocc200'):
                if v=='qsrocc200':self.fake(v,ds,3407 if ds=='agiqa1k' else None)
                runner.verify(v,ds,3407 if ds=='agiqa1k' else None,self.source)
        with patch.object(runner,'control',side_effect=lambda ds,s,f:controls[ds]),contextlib.redirect_stdout(io.StringIO()):
            runner.pack()
            with self.assertRaises(FileExistsError):runner.pack()
            with self.assertRaises(FileExistsError):runner.run('qsrocc050','agiqa1k',3407,True)
        with tarfile.open(self.root/f'{runner.VERSION}_1ksplit3407_results.tar.gz') as t:
            names=t.getnames();self.assertIn('source_fingerprint.json',names);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,_=runner.paths('qsrocc200','agiqa1k',3407);(d/'console.log').write_text('tampered')
        with self.assertRaises(ValueError):runner.verify('qsrocc200','agiqa1k',3407,self.source)


    def test_effective_optimizer_lr_mismatch_rejected(self):
        d,_=self.fake('qsrocc050','agiqa3k',None)
        meta=runner.base.read_json(d/'run_metadata.json')
        next(g for g in meta['optimizer_initialization'] if g['name']=='quality_adapter')['lr']=2e-4
        (d/'run_metadata.json').write_text(json.dumps(meta))
        done=runner.base.read_json(d/'completion_manifest.json')
        done['files_sha256']['run_metadata.json']=runner.base.digest(d/'run_metadata.json')
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'optimizer group'):runner.verify('qsrocc050','agiqa3k',None,self.source)


    def test_six_run_dispatch_and_no_launch_on_preflight_failure(self):
        with patch.object(runner,'control'),patch.object(runner,'run') as launch:
            runner.run_all()
            self.assertEqual([call.args for call in launch.call_args_list],[(v,ds) for ds in runner.DATASETS for v in runner.VARIANTS])
        with patch.object(runner,'control'),patch.object(runner,'run',side_effect=RuntimeError('stop')) as launch:
            with self.assertRaises(RuntimeError):runner.run_all()
            self.assertEqual(launch.call_count,1)
        runner.paths('qsrocc200','aigciqa2023')[0].mkdir(parents=True)
        with patch.object(runner,'control'),patch.object(runner,'run') as launch:
            with self.assertRaises(FileExistsError):runner.run_all()
            launch.assert_not_called()

    def test_effective_loss_weight_mismatch_rejected(self):
        d,table=self.fake('qsrocc050','agiqa3k',None)
        with table.open() as f:rows=list(csv.DictReader(f))
        record=next(r for r in rows if r['event']=='epoch')
        m=json.loads(record['metrics_json']);m['train_train_weighted_quality_srocc']=.99;record['metrics_json']=json.dumps(m)
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader();writer.writerows(rows)
        done=runner.base.read_json(d/'completion_manifest.json');done['csv_sha256']=runner.base.digest(table)
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'loss weight mismatch'):runner.verify('qsrocc050','agiqa3k',None,self.source)

    def test_amp_rounding_in_recorded_rank_is_accepted(self):
        d,table=self.fake('qsrocc050','agiqa3k',None)
        with table.open() as f:rows=list(csv.DictReader(f))
        record=next(r for r in rows if r['event']=='epoch')
        m=json.loads(record['metrics_json']);m['train_train_weighted_quality_rank']*=1.0005;record['metrics_json']=json.dumps(m)
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader();writer.writerows(rows)
        done=runner.base.read_json(d/'completion_manifest.json');done['csv_sha256']=runner.base.digest(table)
        (d/'completion_manifest.json').write_text(json.dumps(done))
        runner.verify('qsrocc050','agiqa3k',None,self.source)

class ActualLossWeightTests(unittest.TestCase):
    def test_only_soft_srocc_objective_changes_in_real_loss_function(self):
        from train_cm_iqa import compute_losses
        from test_downstream_pipeline_controls import ToyScores,batch
        torch.manual_seed(42);model=ToyScores();data=batch()
        cfg=dict(train=dict(two_output_only=True,quality_only=True,lambda_quality=1.,lambda_align=0.,lambda_align_aux=0.,
            lambda_final=0.,lambda_neg=0.,lambda_rank_quality=.03,lambda_srocc_quality=.03,srocc_tau=.02),dataset={})
        _,base_loss=compute_losses(model,data,'cpu',cfg)
        reference=float(base_loss['loss'].detach());raw=float(base_loss['srocc_quality'])
        weight=model.weight.detach().clone()
        for mult in (.5,2.):
            c=copy.deepcopy(cfg);c['train']['lambda_srocc_quality']*=mult
            _,loss=compute_losses(model,data,'cpu',c)
            expected=(mult-1)*cfg['train']['lambda_srocc_quality']*raw
            self.assertAlmostEqual(float(loss['loss'].detach())-reference,expected,places=6)
            self.assertAlmostEqual(float(loss['weighted_quality_srocc']),c['train']['lambda_srocc_quality']*raw,places=6)
            for key in ('quality','rank_quality','srocc_quality','weighted_quality_rank','weighted_quality_mse'):
                torch.testing.assert_close(loss[key],base_loss[key],rtol=0,atol=0)
            gradient=torch.autograd.grad(loss['loss'],model.weight)[0]
            self.assertTrue(torch.isfinite(gradient).all());torch.testing.assert_close(model.weight,weight)
