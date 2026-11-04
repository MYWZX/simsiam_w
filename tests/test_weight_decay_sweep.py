"""No real training: complete transfer, scalar override, provenance and packaging."""
import contextlib
import copy
import csv
import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import yaml
import test_projection_only as projection
from test_projection_ablations import make
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import build_optimizer
from models.two_stage_models import load_pretrained_backbone
from utils.qa_residual_tuning import apply_post_load_qa_alpha
from scripts import run_weight_decay as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        shutil.copytree(runner.ROOT/'configs',self.root/'configs')
        (self.root/'docs').mkdir()
        for n in ['current_baseline.json','current_baseline.csv']:shutil.copy(runner.ROOT/'docs'/n,self.root/'docs'/n)
        self.pin=json.loads((self.root/'docs/current_baseline.json').read_text())['source']
        p=patch.object(runner,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        p=patch.object(runner.base,'completed',return_value=self.pin);p.start();self.addCleanup(p.stop)

    def fake_completed(self,v,ds):
        c=runner.config(v,ds);d,table=runner.paths(v,ds);d.mkdir(parents=True)
        runner.base.write_new(d/'source_fingerprint.json',self.pin)
        runner.base.write_new(d/'launch_manifest.json',dict(version=runner.VERSION,variant=v,dataset=ds,
            config_sha256=runner.base.digest(self.root/f'configs/weight_decay_sweep/{v}/{ds}.yaml'),source=self.pin))
        (d/'config_snapshot.yaml').write_text(yaml.safe_dump(c));(d/'console.log').write_text('fake log')
        runner.base.write_new(d/'run_metadata.json',dict(checkpoint_verified_after_load=True,
            checkpoint={'sha256':self.pin['sha256']}))
        with table.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','metrics_json']);w.writeheader()
            for event,m in [('final',dict(selection_metric='val_quality_SROCC',last_epoch=60))]:w.writerow(dict(event=event,metrics_json=json.dumps(m)))
        names=('config_snapshot.yaml','run_metadata.json','launch_manifest.json','source_fingerprint.json','console.log')
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in names},csv_sha256=runner.base.digest(table)))
        return d

    def test_six_configs_pinned_source_and_no_train_dryrun(self):
        for v in runner.VARIANTS:
            for ds in runner.DATASETS:
                cfg=runner.config(v,ds)
                self.assertEqual(cfg['train']['layerwise_lr']['shared_projection_mult'],1.)
                self.assertNotIn('post_load_qa_alpha',cfg['train'])
                self.assertEqual(cfg['train']['weight_decay'],runner.VARIANTS[v])
                with contextlib.redirect_stdout(io.StringIO()):runner.run(v,ds,True)
                self.assertFalse(runner.paths(v,ds)[0].exists())
        with patch.object(runner.base,'completed',return_value={**self.pin,'sha256':'wrong'}):
            with self.assertRaises(ValueError):runner.source()
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):runner.run('wd005','agiqa3k')
        self.assertFalse(runner.paths('wd005','agiqa3k')[0].exists())

    def test_completed_provenance_pack_and_overwrite_rejection(self):
        for v in runner.VARIANTS:
            for ds in runner.DATASETS:
                self.fake_completed(v,ds);runner.verify(v,ds,self.pin)
        # Synthetic baseline artifacts, never launch the completed control.
        basepaths={}
        for stage in runner.base.STAGES:
            d=self.root/'outputs'/('control_'+stage);d.mkdir();table=d.with_suffix('.csv');table.write_text('baseline evidence')
            (d/'console.log').write_text('baseline log');basepaths[stage]=(d,table)
        with patch.object(runner.base,'paths',side_effect=lambda s:basepaths[s]):
            with contextlib.redirect_stdout(io.StringIO()):runner.pack()
            with self.assertRaises(FileExistsError):runner.pack()
        with tarfile.open(self.root/f'{runner.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertIn('source_fingerprint.json',names)
            self.assertFalse(any(x.endswith(('.pth','.pt','.bin')) for x in names))
        with self.assertRaises(FileExistsError):runner.run('wd005','agiqa3k',True)
        d=runner.paths('wd005','agiqa3k')[0];(d/'console.log').write_text('changed')
        with self.assertRaises(ValueError):runner.verify('wd005','agiqa3k',self.pin)

    def test_other_hyperparameter_change_rejected(self):
        p=self.root/'configs/weight_decay_sweep/wd005/agiqa1k.yaml'
        c=yaml.safe_load(p.read_text());c['train']['lr']=.01;p.write_text(yaml.safe_dump(c))
        with self.assertRaises(ValueError):runner.config('wd005','agiqa1k')


class OptimizerDecayTests(unittest.TestCase):
    def test_all_groups_change_decay_only_with_inherited_alpha(self):
        cfg=projection.config(True)
        cfg['train'].update(lr=1e-4,weight_decay=.002,layerwise_lr={'enabled':True,'fusion_mult':.1,'shared_projection_mult':1.})
        model=make(cfg).iqa
        before={k:v.clone() for k,v in model.state_dict().items()}
        original=build_optimizer(cfg,model)
        groups={g['name']:g for g in original.param_groups}
        for decay in (.005,.01):
            candidate=copy.deepcopy(cfg);candidate['train']['weight_decay']=decay
            opt=build_optimizer(candidate,model)
            self.assertEqual({g['name'] for g in opt.param_groups},set(groups))
            for g in opt.param_groups:
                old=groups[g['name']]
                self.assertEqual(g['weight_decay'],decay)
                self.assertEqual(g['lr'],old['lr'])
                self.assertEqual([id(p) for p in g['params']],[id(p) for p in old['params']])
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertEqual(groups['fusion']['lr'],1e-5)
        self.assertEqual(groups['shared_projection']['lr'],1e-4)
