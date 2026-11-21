"""Alignment-only config/optimizer/loss and artifact checks; synthetic data only."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from scripts import run_align_tuning as r
from scripts.audit_align_selection import audit_csv
from utils.refit_protocol import digest
from utils.run_metadata import optimizer_initialization
from train_cm_iqa import build_optimizer,compute_losses
from test_projection_only import config as tiny
from test_frozen_clip_baseline import small_model
from test_downstream_pipeline_controls import ToyScores,batch


class ConfigAndMathTests(unittest.TestCase):
    def test_eight_configs_have_exactly_one_hyperparameter_change(self):
        paths=[]
        for ds in r.DATASETS:
            base=yaml.safe_load((r.ROOT/f'configs/current_tuning_reference/{ds}.yaml').read_text())
            for v,(kind,value) in r.VARIANTS.items():
                c=r.config(v,ds);expected=copy.deepcopy(base)
                for key in ('output_dir','results_csv','run_id'):expected[key]=c[key]
                expected['train'].update(record_split_manifest=True,record_optimizer_initialization=True)
                if kind=='lr':expected['train']['layerwise_lr']['align_mult']=value
                else:expected['train']['lambda_srocc_align']=value
                self.assertEqual(c,expected)
                self.assertEqual(c['model']['nss_text_gate']['nss_dim'],6)
                self.assertEqual(c['train']['layerwise_lr']['shared_projection_mult'],1.)
                paths.append(c['output_dir'])
        self.assertEqual(len(set(paths)),8)
        for v,ds,seed in [('bad','agiqa3k',None),('alignlr050','agiqa1k',None),('asrocc050','aigciqa2023',3407)]:
            with self.assertRaises(ValueError):r.config(v,ds,seed)

    def test_actual_optimizer_only_align_group_lr_changes_and_all_parameter_memberships_match(self):
        saved=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,saved);torch.manual_seed(42)
        c=tiny(True);c['train']=copy.deepcopy(r.config('asrocc050','agiqa3k')['train'])
        c['train']['lambda_srocc_align']=.04
        c['model']['align_adapter']=copy.deepcopy(r.config('asrocc050','agiqa3k')['model']['align_adapter'])
        m=small_model(c).eval();base=build_optimizer(c,m)
        groups={g['name']:{id(p) for p in g['params']} for g in base.param_groups}
        self.assertIn(id(m.align_global_adapter.net[1].weight),groups['align'])
        self.assertIn(id(m.align_patch_adapter.net[1].weight),groups['align'])
        self.assertIn(id(m.align_text_adapter.net[1].weight),groups['align'])
        self.assertIn(id(m.text_to_fused.weight),groups['align'])
        for v in r.VARIANTS:
            changed=copy.deepcopy(c);changed['train']=r.config(v,'agiqa3k')['train'];opt=build_optimizer(changed,m)
            for g in opt.param_groups:
                self.assertEqual({id(p) for p in g['params']},groups[g['name']])
                self.assertAlmostEqual(g['lr'],r.expected_group_lrs(changed)[g['name']])
            ownership={id(p):g for g in opt.param_groups for p in g['params']}
            self.assertEqual(ownership[id(m.shared_image_projector.net[0].weight)]['lr'],1e-4)
            self.assertEqual(ownership[id(m.shared_text_projector.net[0].weight)]['lr'],1e-4)
            self.assertEqual(ownership[id(m.backbone.fusion.semantic_proj.weight)]['lr'],1e-5)
            self.assertEqual(ownership[id(m.backbone.text_encoder.context_vectors)]['lr'],1e-5)

    def test_actual_loss_and_gradient_change_only_by_alignment_srocc_coefficient(self):
        saved=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,saved);torch.manual_seed(42)
        model=ToyScores();data=batch();data['image']=torch.tensor([-.8,-.1,.15,.5])[:,None,None,None].expand(-1,3,8,8).clone()
        data['align_score']=torch.tensor([.15,.65,.4,.85])
        c=r.config('alignlr050','agiqa3k');out,base=compute_losses(model,data,'cpu',c)
        for value in (.02,.08):
            cc=copy.deepcopy(c);cc['train']['lambda_srocc_align']=value
            changed,loss=compute_losses(model,data,'cpu',cc)
            delta=(value-.04)*base['srocc_align']
            torch.testing.assert_close(loss['loss']-base['loss'],delta,atol=1e-7,rtol=1e-5)
            torch.testing.assert_close(loss['loss_quality_task'],base['loss_quality_task'],rtol=0,atol=0)
            for k in ('quality','align','rank_quality','rank_align','srocc_quality','srocc_align'):
                torch.testing.assert_close(loss[k],base[k],rtol=0,atol=0)
            for k in out:
                if isinstance(out[k],torch.Tensor):torch.testing.assert_close(out[k],changed[k],rtol=0,atol=0)
            gradient=torch.autograd.grad(loss['loss'],model.weight)[0];self.assertTrue(torch.isfinite(gradient).all())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        shutil.copytree(r.ROOT/'configs/current_tuning_reference',self.root/'configs/current_tuning_reference')
        (self.root/'docs').mkdir()
        for name in ('current_tuning_reference.json','current_tuning_reference.md','align_tuning.md'):
            shutil.copy(r.ROOT/'docs'/name,self.root/'docs'/name)
        p=patch.object(r,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.source={'sha256':'pinned'};p=patch.object(r,'source',return_value=self.source);p.start();self.addCleanup(p.stop)

    def fake(self,v,ds,complete=True):
        c=r.config(v,ds);d,t=r.paths(v,ds);d.mkdir(parents=True,exist_ok=True)
        for name in ('launch_config.yaml','config_snapshot.yaml'):
            if not (d/name).exists():(d/name).write_text(yaml.safe_dump(c))
        if not (d/'console.log').exists():(d/'console.log').write_text('fake')
        if not (d/'source_fingerprint.json').exists():r.base.write_new(d/'source_fingerprint.json',self.source)
        if not (d/'launch_manifest.json').exists():r.base.write_new(d/'launch_manifest.json',dict(version=r.VERSION,variant=v,dataset=ds,split_seed=42,source=self.source,config_sha256=r.base.digest(d/'launch_config.yaml')))
        sizes=dict(train=7,val=1,test=2);order=torch.randperm(10,generator=torch.Generator().manual_seed(42)).tolist()
        indices=dict(train=order[:7],val=order[7:8],test=order[8:]);evidence=dict(train_seed=42,split_seed=42,indices=indices,split_sizes=sizes,split_indices_sha256=digest(indices),dataset_ids_sha256='dataset')
        r.base.write_new(d/'data_split_manifest.json',evidence)
        groups=[dict(name=n,lr=lr,weight_decay=.002,parameter_count=1) for n,lr in r.expected_group_lrs(c).items()]
        r.base.write_new(d/'run_metadata.json',dict(seed=42,split_seed=42,checkpoint_verified_after_load=True,checkpoint=self.source,optimizer_initialization=groups,data_split={k:v for k,v in evidence.items() if k!='indices'}))
        metrics=dict(train_train_quality=.02,train_train_align=.03,train_train_align_global=.04,train_train_align_patch=.05,
            train_train_rank_quality=.1,train_train_rank_align=.2,train_train_srocc_quality=.3,train_train_srocc_align=.4,train_train_pert=.01)
        tc=c['train'];metrics.update(train_train_weighted_quality_mse=.02,train_train_weighted_quality_rank=.1*tc['lambda_rank_quality'],train_train_weighted_quality_srocc=.3*tc['lambda_srocc_quality'])
        metrics['train_train_loss']=.02+.03+.09*tc['lambda_align_aux']+.1*tc['lambda_rank_quality']+.2*tc['lambda_rank_align']+.3*tc['lambda_srocc_quality']+.4*tc['lambda_srocc_align']+.01*tc['lambda_pert']
        bt={'test_quality_SROCC':.8,'test_align_SROCC':.7}
        with t.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            w.writerow(dict(event='optimizer_initialization',epoch=0,metrics_json=json.dumps({'groups':groups})))
            for e in range(1,61):
                m=dict(metrics,val_val_quality_SROCC=1-abs(e-40)/100,val_val_align_SROCC=1-abs(e-30)/100)
                w.writerow(dict(event='epoch',epoch=e,metrics_json=json.dumps(m)))
            w.writerow(dict(event='best_val',epoch=40,metrics_json='{}'))
            w.writerow(dict(event='best_test',epoch=40,metrics_json=json.dumps(bt)))
            w.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,best_epoch=40,selection_metric='val_quality_SROCC',best_test=bt))))
        if complete:self.seal(d,t)
        (d/'unused_weights.pth').write_bytes(b'excluded')
        return d,t

    def seal(self,d,t):
        (d/'completion_manifest.json').write_text(json.dumps(dict(files_sha256={n:r.base.digest(d/n) for n in r.FILES if n!='completion_manifest.json'},csv_sha256=r.base.digest(t))))

    def test_pack_all_eight_and_refuse_changed_artifacts_and_overwrite(self):
        controls={}
        for ds in r.DATASETS:
            for v in r.VARIANTS:
                d,t=self.fake(v,ds);r.verify(v,ds,None,self.source)
                controls.setdefault(ds,(d,t))
        with patch.object(r,'control',side_effect=lambda ds,s,fp:controls[ds]),contextlib.redirect_stdout(io.StringIO()):
            r.pack()
            with self.assertRaises(FileExistsError):r.pack()
        with tarfile.open(self.root/f'{r.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertEqual(sum(x.endswith('.csv') for x in names),8)
            self.assertFalse(any(x.endswith('.pth') for x in names));self.assertIn('source_fingerprint.json',names)
        d,_=r.paths('alignlr050','agiqa3k');(d/'console.log').write_text('changed')
        with self.assertRaises(ValueError):r.verify('alignlr050','agiqa3k',None,self.source)

    def test_optimizer_loss_and_selection_tampering_rejected_even_with_updated_hashes(self):
        for case in ('lr','objective','selection'):
            with self.subTest(case=case):
                ds=r.DATASETS[0];v=list(r.VARIANTS)[('lr','objective','selection').index(case)];d,t=self.fake(v,ds)
                if case=='lr':
                    meta=r.base.read_json(d/'run_metadata.json');next(g for g in meta['optimizer_initialization'] if g['name']=='align')['lr']=1e-9;(d/'run_metadata.json').write_text(json.dumps(meta))
                else:
                    with t.open() as f:rows=list(csv.DictReader(f))
                    row=next(x for x in rows if x['event']==('epoch' if case=='objective' else 'best_test'))
                    if case=='objective':
                        m=json.loads(row['metrics_json']);m['train_train_loss']+=.1;row['metrics_json']=json.dumps(m)
                    else:row['epoch']='30'
                    with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
                self.seal(d,t)
                with self.assertRaises(ValueError):r.verify(v,ds,None,self.source)

    def test_queue_preflight_dry_run_failure_and_continue_completed(self):
        for queue in ('lr','srocc','all'):
            with patch.object(r,'control'),patch.object(r,'run') as launch:r.run_all(queue)
            vs=tuple(r.VARIANTS) if queue=='all' else r.QUEUES[queue]
            self.assertEqual([x.args for x in launch.call_args_list],[(v,ds) for ds in r.DATASETS for v in vs])
        with patch.object(r,'control'),patch.object(r,'run',side_effect=RuntimeError('trainer failed')) as launch:
            with self.assertRaises(RuntimeError):r.run_all('all')
            self.assertEqual(launch.call_count,1)
        with patch.object(r,'source',side_effect=AssertionError('dry plan must not load weights')),patch.object(r,'run',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):r.run_all('all',True)
        d,t=self.fake('alignlr050','agiqa3k')
        with patch.object(r,'control'),patch.object(r,'run') as launch:
            with self.assertRaises(FileExistsError):r.run_all('lr')
            launch.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):r.run_all('lr',continue_completed=True)
            self.assertEqual(launch.call_count,3)
        partial,_=r.paths('alignlr200','aigciqa2023');partial.mkdir(parents=True)
        with patch.object(r,'control'),patch.object(r,'run') as launch:
            with self.assertRaises(FileNotFoundError):r.run_all('lr',continue_completed=True)
            launch.assert_not_called()
        self.assertTrue(t.exists())

    def test_cpu_refusal_fake_trainer_and_real_trainer_entry_is_downstream_only(self):
        with patch.object(r,'control'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('alignlr050','agiqa3k')
        self.assertFalse(r.paths('alignlr050','agiqa3k')[0].exists())
        outer=self
        class Process:
            def __init__(self,command,**kw):
                outer.assertEqual(command[1],'train_cm_iqa.py');outer.fake('alignlr050','agiqa3k',False);self.stdout=iter(['fake downstream\n'])
            def wait(self):return 0
        with patch.object(r,'control'),patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Process),contextlib.redirect_stdout(io.StringIO()):r.run('alignlr050','agiqa3k')
        r.verify('alignlr050','agiqa3k',None,self.source)

    def test_wrong_config_source_budget_and_nss_refused_before_launch(self):
        path=self.root/'configs/current_tuning_reference/agiqa3k.yaml'
        original=yaml.safe_load(path.read_text())
        for kind in ('source','budget','nss'):
            c=copy.deepcopy(original)
            if kind=='source':c['train']['pretrained_checkpoint']='outputs/other/best.pth'
            elif kind=='budget':c['train']['epochs']=30
            else:c['model']['nss_text_gate'].update(descriptor='mscn_stats5_v1',nss_dim=5)
            path.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):r.config('alignlr050','agiqa3k')

    def test_pinned_source_rejects_other_pretraining(self):
        # source() itself is mocked in fixture; use a fresh module for resolver.
        import importlib.util
        spec=importlib.util.spec_from_file_location('_align_source',Path(__file__).resolve().parents[1]/'scripts/run_align_tuning.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=self.root
        source=module.reference()['source']
        with patch.object(module.splits,'source',return_value=source):self.assertEqual(module.source(),source)
        bad=dict(source,sha256='wrong')
        with patch.object(module.splits,'source',return_value=bad),self.assertRaises(ValueError):module.source()

    def test_selection_audit_uses_validation_only_and_preserves_recorded_selection(self):
        _,t=self.fake('alignlr050','agiqa3k');report=audit_csv(t)
        self.assertEqual(report['quality_selected_epoch'],40);self.assertEqual(report['align_validation_best_epoch'],30)
        self.assertGreater(report['val_align_gap'],0)
        self.assertFalse(any(k.startswith('test') for k in report))

if __name__=='__main__':unittest.main()
