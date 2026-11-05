"""CLIP-only LR changes with fixed source/splits; no real training."""
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
from scripts import run_clip_lr as runner

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        shutil.copytree(runner.ROOT/'configs/projection_only_v1_baseline',self.root/'configs/projection_only_v1_baseline')
        (self.root/'docs').mkdir();(self.root/'docs/current_baseline.json').write_text('{}');(self.root/'docs/clip_lr.md').write_text('protocol')
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
            writer=csv.DictWriter(f,fieldnames=['event','metrics_json']);writer.writeheader()
            for event,m in [('optimizer_initialization',{'groups':record}),('final',dict(last_epoch=60,selection_metric='val_quality_SROCC'))]:writer.writerow(dict(event=event,metrics_json=json.dumps(m)))
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in runner.FILES if n!='completion_manifest.json'},csv_sha256=runner.base.digest(table)))
        (d/'unused_weights.pth').write_bytes(b'excluded')
        return d,table

    def test_fixed_splits_config_and_cpu_refusal(self):
        with self.assertRaises(ValueError):runner.config('glr001','agiqa1k',3047)
        with self.assertRaises(ValueError):runner.config('glr001','agiqa3k',3407)
        for ds in runner.DATASETS:
            original=yaml.safe_load((self.root/f'configs/projection_only_v1_baseline/{ds}.yaml').read_text())
            for v in runner.VARIANTS:
                c=runner.config(v,ds)
                self.assertEqual(c['seed'],42);self.assertEqual(c['split_seed'],3407 if ds=='agiqa1k' else 42)
                self.assertEqual(c['model'],original['model'])
                self.assertEqual(c['train']['weight_decay'],.002)
                self.assertNotIn('post_load_qa_alpha',c['train'])
                expected=dict(original['train']['layerwise_lr']);expected['global_encoder_mult']=runner.VARIANTS[v]
                self.assertEqual(c['train']['layerwise_lr'],expected)
        with patch.object(runner,'control',side_effect=ValueError('missing control')):
            with self.assertRaisesRegex(ValueError,'missing control'):runner.run('glr001','agiqa1k',dry_run=True)
        with patch.object(runner,'control'),contextlib.redirect_stdout(io.StringIO()):runner.run('glr001','agiqa3k',dry_run=True)
        with patch.object(runner,'control'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):runner.run('glr001','agiqa3k')

    def test_control_provenance_pack_and_no_overwrite(self):
        controls={}
        for ds in runner.DATASETS:controls[ds]=self.fake('glr001',ds,3407 if ds=='agiqa1k' else None)
        # Replace their control labels only in the mocked control resolver; reuse
        # identical actual memberships to exercise package checks.
        for ds in runner.DATASETS:
            for v in ('glr001','glr004'):
                if v=='glr004':self.fake(v,ds,3407 if ds=='agiqa1k' else None)
                runner.verify(v,ds,3407 if ds=='agiqa1k' else None,self.source)
        with patch.object(runner,'control',side_effect=lambda ds,s,f:controls[ds]),contextlib.redirect_stdout(io.StringIO()):
            runner.pack()
            with self.assertRaises(FileExistsError):runner.pack()
            with self.assertRaises(FileExistsError):runner.run('glr001','agiqa1k',3407,True)
        with tarfile.open(self.root/f'{runner.VERSION}_1ksplit3407_results.tar.gz') as t:
            names=t.getnames();self.assertIn('source_fingerprint.json',names);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,_=runner.paths('glr004','agiqa1k',3407);(d/'console.log').write_text('tampered')
        with self.assertRaises(ValueError):runner.verify('glr004','agiqa1k',3407,self.source)


    def test_effective_optimizer_lr_mismatch_rejected(self):
        d,_=self.fake('glr001','agiqa3k',None)
        meta=runner.base.read_json(d/'run_metadata.json')
        meta['optimizer_initialization'][0]['lr']=2e-6
        (d/'run_metadata.json').write_text(json.dumps(meta))
        done=runner.base.read_json(d/'completion_manifest.json')
        done['files_sha256']['run_metadata.json']=runner.base.digest(d/'run_metadata.json')
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'optimizer group'):runner.verify('glr001','agiqa3k',None,self.source)

class OptimizerTests(unittest.TestCase):
    def test_only_global_lr_changes_membership_weights_and_other_groups_unchanged(self):
        cfg=projection.config(True)
        cfg['train'].update(lr=1e-4,weight_decay=.002,layerwise_lr={'enabled':True,'global_encoder_mult':.02,'local_encoder_mult':.02,'fusion_mult':.1,'shared_projection_mult':1.})
        model=make(cfg).iqa;weights={k:v.clone() for k,v in model.state_dict().items()}
        original=build_optimizer(cfg,model);groups={g['name']:g for g in original.param_groups}
        for mult in (.01,.04):
            c=copy.deepcopy(cfg);c['train']['layerwise_lr']['global_encoder_mult']=mult
            opt=build_optimizer(c,model);record=optimizer_initialization(opt)
            self.assertEqual({g['name'] for g in record},set(groups))
            for g in opt.param_groups:
                old=groups[g['name']]
                self.assertEqual([id(p) for p in g['params']],[id(p) for p in old['params']])
                self.assertEqual(g['weight_decay'],old['weight_decay'])
                self.assertEqual(g['lr'],1e-4*mult if g['name']=='global_encoder' else old['lr'])
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-6)
            rates=[]
            for epoch in range(60):
                opt.step();sched.step();rates.append(next(g['lr'] for g in opt.param_groups if g['name']=='global_encoder'))
            self.assertAlmostEqual(rates[-1],1e-6)
            if mult==.01:self.assertTrue(all(math.isclose(x,1e-6) for x in rates))
            for k,v in model.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=0,atol=0)
