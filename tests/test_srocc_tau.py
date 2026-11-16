"""Shared Soft-SROCC temperature control from the original v2 six-channel baseline; no real training."""
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
from scripts import run_srocc_tau as runner

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        shutil.copytree(runner.ROOT/'configs/current_tuning_reference',self.root/'configs/current_tuning_reference')
        (self.root/'docs').mkdir();(self.root/'docs/current_baseline.json').write_text('{}');(self.root/'docs/srocc_tau.md').write_text('protocol')
        for name in ('current_tuning_reference.json','current_tuning_reference.md'):
            shutil.copy(runner.ROOT/'docs'/name,self.root/'docs'/name)
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
        with self.assertRaises(ValueError):runner.config('tau010','agiqa1k',3047)
        with self.assertRaises(ValueError):runner.config('tau010','agiqa3k',3407)
        for ds in runner.DATASETS:
            original=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            for v in runner.VARIANTS:
                c=runner.config(v,ds)
                self.assertEqual(c['seed'],42);self.assertEqual(c['split_seed'],3407 if ds=='agiqa1k' else 42)
                self.assertEqual(c['model'],original['model'])
                self.assertEqual(c['train']['lambda_srocc_quality'],{'agiqa1k':.03,'agiqa3k':.02,'aigciqa2023':.04}[ds])
                self.assertEqual(c['train']['lambda_rank_quality'],.03)
                self.assertEqual(c['train']['srocc_tau'],runner.VARIANTS[v])
                self.assertEqual(c['train']['weight_decay'],.002)
                self.assertNotIn('post_load_qa_alpha',c['train'])
                expected=dict(original['train']['layerwise_lr']);expected=dict(original['train']['layerwise_lr'])
                self.assertEqual(c['train']['layerwise_lr'],expected)
        with patch.object(runner,'control',side_effect=ValueError('missing control')):
            with self.assertRaisesRegex(ValueError,'missing control'):runner.run('tau010','agiqa1k',dry_run=True)
        with patch.object(runner,'control'),contextlib.redirect_stdout(io.StringIO()):runner.run('tau010','agiqa3k',dry_run=True)
        with patch.object(runner,'control'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):runner.run('tau010','agiqa3k')

    def test_config_diff_is_tau_and_evidence_only_and_new_nss_is_refused(self):
        for ds in runner.DATASETS:
            original=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            for v in runner.VARIANTS:
                actual=runner.config(v,ds);expected=copy.deepcopy(original)
                for k in ('output_dir','results_csv','run_id'):expected[k]=actual[k]
                expected['train'].update(srocc_tau=runner.VARIANTS[v],record_split_manifest=True,record_optimizer_initialization=True)
                self.assertEqual(actual,expected)
        with self.assertRaises(ValueError):runner.config('unknown','agiqa1k')
        p=self.root/'configs/current_tuning_reference/agiqa1k.yaml';c=yaml.safe_load(p.read_text())
        c['model']['nss_text_gate'].update(descriptor='legacy_stats5_v1',nss_dim=5);p.write_text(yaml.safe_dump(c))
        with self.assertRaisesRegex(ValueError,'six-dimensional'):runner.config('tau010','agiqa1k')

    def test_pinned_original_source_is_required(self):
        # Temporarily restore the real source resolver hidden by fixture mock.
        import importlib.util
        spec=importlib.util.spec_from_file_location('_tau_source_test',Path(__file__).resolve().parents[1]/'scripts/run_srocc_tau.py')
        actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual);actual.ROOT=self.root
        pinned=actual.reference()['source']
        with patch.object(actual.splits,'source',return_value=copy.deepcopy(pinned)):
            self.assertEqual(actual.source(),pinned)
        wrong=copy.deepcopy(pinned);wrong['sha256']='different-nss5-source'
        with patch.object(actual.splits,'source',return_value=wrong):
            with self.assertRaisesRegex(ValueError,'differs from v2'):actual.source()

    def test_control_provenance_pack_and_no_overwrite(self):
        controls={}
        for ds in runner.DATASETS:controls[ds]=self.fake('tau010',ds,3407 if ds=='agiqa1k' else None)
        # Replace their control labels only in the mocked control resolver; reuse
        # identical actual memberships to exercise package checks.
        for ds in runner.DATASETS:
            for v in ('tau010','tau040'):
                if v=='tau040':self.fake(v,ds,3407 if ds=='agiqa1k' else None)
                runner.verify(v,ds,3407 if ds=='agiqa1k' else None,self.source)
        with patch.object(runner,'control',side_effect=lambda ds,s,f:controls[ds]),contextlib.redirect_stdout(io.StringIO()):
            runner.pack()
            with self.assertRaises(FileExistsError):runner.pack()
            with self.assertRaises(FileExistsError):runner.run('tau010','agiqa1k',3407,True)
        with tarfile.open(self.root/f'{runner.VERSION}_1ksplit3407_results.tar.gz') as t:
            names=t.getnames();self.assertIn('source_fingerprint.json',names);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,_=runner.paths('tau040','agiqa1k',3407);(d/'console.log').write_text('tampered')
        with self.assertRaises(ValueError):runner.verify('tau040','agiqa1k',3407,self.source)


    def test_effective_optimizer_lr_mismatch_rejected(self):
        d,_=self.fake('tau010','agiqa3k',None)
        meta=runner.base.read_json(d/'run_metadata.json')
        next(g for g in meta['optimizer_initialization'] if g['name']=='quality_adapter')['lr']=2e-4
        (d/'run_metadata.json').write_text(json.dumps(meta))
        done=runner.base.read_json(d/'completion_manifest.json')
        done['files_sha256']['run_metadata.json']=runner.base.digest(d/'run_metadata.json')
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'optimizer group'):runner.verify('tau010','agiqa3k',None,self.source)


    def test_six_run_dispatch_and_no_launch_on_preflight_failure(self):
        with patch.object(runner,'control'),patch.object(runner,'run') as launch:
            runner.run_all()
            self.assertEqual([call.args for call in launch.call_args_list],[(v,ds) for ds in runner.DATASETS for v in runner.VARIANTS])
        with patch.object(runner,'control'),patch.object(runner,'run',side_effect=RuntimeError('stop')) as launch:
            with self.assertRaises(RuntimeError):runner.run_all()
            self.assertEqual(launch.call_count,1)
        runner.paths('tau040','aigciqa2023')[0].mkdir(parents=True)
        with patch.object(runner,'control'),patch.object(runner,'run') as launch:
            with self.assertRaises(FileExistsError):runner.run_all()
            launch.assert_not_called()

    def test_effective_loss_weight_mismatch_rejected(self):
        d,table=self.fake('tau010','agiqa3k',None)
        with table.open() as f:rows=list(csv.DictReader(f))
        record=next(r for r in rows if r['event']=='epoch')
        m=json.loads(record['metrics_json']);m['train_train_weighted_quality_srocc']=.99;record['metrics_json']=json.dumps(m)
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader();writer.writerows(rows)
        done=runner.base.read_json(d/'completion_manifest.json');done['csv_sha256']=runner.base.digest(table)
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'loss weight mismatch'):runner.verify('tau010','agiqa3k',None,self.source)

    def test_amp_rounding_in_recorded_rank_is_accepted(self):
        d,table=self.fake('tau010','agiqa3k',None)
        with table.open() as f:rows=list(csv.DictReader(f))
        record=next(r for r in rows if r['event']=='epoch')
        m=json.loads(record['metrics_json']);m['train_train_weighted_quality_rank']*=1.0005;record['metrics_json']=json.dumps(m)
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader();writer.writerows(rows)
        done=runner.base.read_json(d/'completion_manifest.json');done['csv_sha256']=runner.base.digest(table)
        (d/'completion_manifest.json').write_text(json.dumps(done))
        runner.verify('tau010','agiqa3k',None,self.source)

class ActualTemperatureTests(unittest.TestCase):
    def test_only_soft_rank_terms_change_and_all_tasks_receive_shared_tau(self):
        from train_cm_iqa import compute_losses,masked_soft_spearman_loss
        from test_downstream_pipeline_controls import ToyScores,batch
        rng=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,rng)
        torch.manual_seed(42);model=ToyScores();data=batch()
        data['image']=torch.tensor([-.8,-.1,.15,.5])[:,None,None,None].expand(-1,3,8,8).clone()
        data['align_score']=torch.tensor([.15,.65,.4,.85])
        cfg=dict(train=dict(two_output_only=True,quality_only=False,lambda_quality=1.,lambda_align=1.,lambda_align_aux=0.,
            lambda_final=0.,lambda_neg=0.,lambda_rank_quality=.03,lambda_rank_align=.02,
            lambda_srocc_quality=.03,lambda_srocc_align=.04,lambda_srocc_final=0.,srocc_tau=.02,
            use_hard_rank=True,hard_rank_ratio=.35),dataset={})
        base_output,baseline=compute_losses(model,data,'cpu',cfg)
        changed=[]
        for tau in runner.VARIANTS.values():
            c=copy.deepcopy(cfg);c['train']['srocc_tau']=tau
            with patch('train_cm_iqa.masked_soft_spearman_loss',wraps=masked_soft_spearman_loss) as soft:
                out,loss=compute_losses(model,data,'cpu',c)
            self.assertEqual([x.kwargs['tau'] for x in soft.call_args_list],[tau,tau,tau])
            for key in ('quality','align','rank_quality','rank_align','weighted_quality_mse','weighted_quality_rank'):
                torch.testing.assert_close(loss[key],baseline[key],atol=0,rtol=0)
            for key in out:
                if isinstance(out[key],torch.Tensor):torch.testing.assert_close(out[key],base_output[key],atol=0,rtol=0)
            delta=.03*(loss['srocc_quality']-baseline['srocc_quality'])+.04*(loss['srocc_align']-baseline['srocc_align'])
            torch.testing.assert_close(loss['loss']-baseline['loss'],delta,atol=1e-7,rtol=1e-5)
            grad=torch.autograd.grad(loss['loss'],model.weight)[0];self.assertTrue(torch.isfinite(grad).all())
            changed.append(float(abs(delta)))
        self.assertTrue(all(x>1e-7 for x in changed))


class SelectedControlTests(unittest.TestCase):
    def test_2023_control_uses_completed_double_srocc_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'docs').mkdir();(root/'configs/current_tuning_reference').mkdir(parents=True)
            shutil.copy(runner.ROOT/'configs/current_tuning_reference/aigciqa2023.yaml',root/'configs/current_tuning_reference/aigciqa2023.yaml')
            cfg=yaml.safe_load((root/'configs/current_tuning_reference/aigciqa2023.yaml').read_text())
            d=root/cfg['output_dir'];d.mkdir(parents=True);table=root/cfg['results_csv'];table.write_text('verified chosen control')
            (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
            ref=runner.reference();record=next(x for x in ref['results'] if x['dataset']=='aigciqa2023');record['csv_sha256']=runner.base.digest(table)
            (root/'docs/current_tuning_reference.json').write_text(json.dumps(ref))
            with patch.object(runner,'ROOT',root),patch.object(runner.previous_srocc,'verify') as check,patch.object(runner.previous_srocc,'paths',return_value=(d,table)):
                self.assertEqual(runner.control('aigciqa2023',None,{'sha256':'source'}),(d,table))
                check.assert_called_with('qsrocc200','aigciqa2023',None,{'sha256':'source'})
                old=copy.deepcopy(cfg);old['train']['lambda_srocc_quality']=.02;(d/'config_snapshot.yaml').write_text(yaml.safe_dump(old))
                with self.assertRaises(ValueError):runner.control('aigciqa2023',None,{'sha256':'source'})
                (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg));table.write_text('changed')
                with self.assertRaisesRegex(ValueError,'Control CSV'):runner.control('aigciqa2023',None,{'sha256':'source'})
