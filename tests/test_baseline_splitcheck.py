"""No real training: complete transfer, scalar override, provenance and packaging."""
import contextlib
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
from scripts import run_baseline_splitcheck as runner


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
            config_sha256=runner.base.digest(self.root/f'configs/baseline_splitcheck/{v}/{ds}.yaml'),source=self.pin))
        (d/'config_snapshot.yaml').write_text(yaml.safe_dump(c));(d/'console.log').write_text('fake log')
        from utils.refit_protocol import digest
        sizes = dict(train=755,val=108,test=215) if ds=='agiqa1k' else dict(train=7,val=1,test=2)
        order=torch.randperm(sum(sizes.values()),generator=torch.Generator().manual_seed(runner.VARIANTS[v])).tolist()
        n,va=sizes['train'],sizes['val']
        indices=dict(train=order[:n],val=order[n:n+va],test=order[n+va:])
        split=dict(train_seed=42,split_seed=runner.VARIANTS[v],split_sizes=sizes,indices=indices,split_indices_sha256=digest(indices))
        runner.base.write_new(d/'data_split_manifest.json',split)
        runner.base.write_new(d/'run_metadata.json',dict(checkpoint_verified_after_load=True,
            checkpoint={'sha256':self.pin['sha256']},seed=42,split_seed=runner.VARIANTS[v],data_split={k:v for k,v in split.items() if k!='indices'}))
        with table.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','metrics_json']);w.writeheader()
            for event,m in [('final',dict(selection_metric='val_quality_SROCC',last_epoch=60))]:w.writerow(dict(event=event,metrics_json=json.dumps(m)))
        names=('config_snapshot.yaml','run_metadata.json','launch_manifest.json','source_fingerprint.json','console.log','data_split_manifest.json')
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in names},csv_sha256=runner.base.digest(table)))
        return d

    def test_six_configs_pinned_source_and_no_train_dryrun(self):
        for v in runner.VARIANTS:
            for ds in runner.DATASETS:
                cfg=runner.config(v,ds)
                self.assertEqual(cfg['train']['layerwise_lr']['shared_projection_mult'],1.)
                self.assertNotIn('post_load_qa_alpha',cfg['train'])
                self.assertEqual(cfg['train']['weight_decay'],.002)
                self.assertEqual(cfg['seed'],42)
                self.assertEqual(cfg['split_seed'],runner.VARIANTS[v])
                self.assertIn('_seed42/',cfg['train']['pretrained_checkpoint'])
                with contextlib.redirect_stdout(io.StringIO()):runner.run(v,ds,True)
                self.assertFalse(runner.paths(v,ds)[0].exists())
        with patch.object(runner.base,'completed',return_value={**self.pin,'sha256':'wrong'}):
            with self.assertRaises(ValueError):runner.source()
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):runner.run('split123','agiqa3k')
        self.assertFalse(runner.paths('split123','agiqa3k')[0].exists())

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
        with self.assertRaises(FileExistsError):runner.run('split123','agiqa3k',True)
        d=runner.paths('split123','agiqa3k')[0];(d/'console.log').write_text('changed')
        with self.assertRaises(ValueError):runner.verify('split123','agiqa3k',self.pin)

    def test_other_hyperparameter_change_rejected(self):
        p=self.root/'configs/baseline_splitcheck/split123/agiqa1k.yaml'
        c=yaml.safe_load(p.read_text());c['train']['lr']=.01;p.write_text(yaml.safe_dump(c))
        with self.assertRaises(ValueError):runner.config('split123','agiqa1k')

    def test_runtime_split_mismatch_rejected_even_with_valid_hash(self):
        d=self.fake_completed('split123','agiqa1k')
        meta=runner.base.read_json(d/'run_metadata.json');meta['seed']=123
        (d/'run_metadata.json').write_text(json.dumps(meta))
        done=runner.base.read_json(d/'completion_manifest.json')
        done['files_sha256']['run_metadata.json']=runner.base.digest(d/'run_metadata.json')
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'seed mismatch'):runner.verify('split123','agiqa1k',self.pin)

    def test_split_config_change_rejected(self):
        p=self.root/'configs/baseline_splitcheck/split3407/agiqa3k.yaml'
        c=yaml.safe_load(p.read_text());c['seed']=3407;p.write_text(yaml.safe_dump(c))
        with self.assertRaises(ValueError):runner.config('split3407','agiqa3k')


import test_agiqa1k_subset1078 as original_subset
from datasets.multi_iqa_dataset import build_agiqa1k_iqa_dataloaders
from utils.agiqa1k_subset1078 import RESPLIT_REVISION, expected_resplit_manifest, prepare_resplit_manifest
from utils.split_seed_evidence import record_split

class ActualMembershipTests(unittest.TestCase):
    def setUp(self):original_subset.Subset1078Tests.setUp(self)

    def test_two_real_loader_splits_preserve_samples_counts_and_train_seed(self):
        original=self.manifest_path.read_bytes();seen=[]
        for seed in (123,3407):
            p=prepare_resplit_manifest(self.root,seed)
            cfg=json.loads(json.dumps(self.cfg));cfg['seed']=42;cfg['split_seed']=seed
            cfg['dataset'].update(data_revision=RESPLIT_REVISION,split_manifest_file=str(p.relative_to(self.root)))
            cfg['train']['record_split_manifest']=True
            loaders=build_agiqa1k_iqa_dataloaders(cfg)
            self.assertEqual([len(l.dataset) for l in loaders],[755,108,215])
            expected=expected_resplit_manifest(seed)
            for loader,part in zip(loaders,['train','val','test']):self.assertEqual(list(loader.dataset.indices),expected['split_indices'][part])
            self.assertEqual(set().union(*(set(l.dataset.indices) for l in loaders)),set(range(1078)))
            output=self.root/f'output{seed}';output.mkdir()
            state=torch.random.get_rng_state().clone();evidence=record_split(cfg,loaders,output)
            torch.testing.assert_close(torch.random.get_rng_state(),state)
            self.assertEqual(evidence['train_seed'],42);self.assertEqual(evidence['split_seed'],seed)
            self.assertNotEqual(evidence['split_indices_sha256'],evidence['reference_seed42_indices_sha256'])
            self.assertEqual(cfg['dataset']['fitted_mos_normalization']['fit_count'],755)
            self.assertEqual(cfg['dataset']['fitted_mos_normalization']['fit_source'],'train_indices')
            seen.append(evidence['indices'])
        self.assertNotEqual(seen[0],seen[1]);self.assertEqual(self.manifest_path.read_bytes(),original)

    def test_manifest_tampering_and_missing_manifest_refused(self):
        p=prepare_resplit_manifest(self.root,123)
        cfg=json.loads(json.dumps(self.cfg));cfg['split_seed']=123
        cfg['dataset'].update(data_revision=RESPLIT_REVISION,split_manifest_file=str(p.relative_to(self.root)))
        manifest=json.loads(p.read_text());manifest['split_indices']['train'].reverse();p.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):prepare_resplit_manifest(self.root,123)
        with self.assertRaises(ValueError):build_agiqa1k_iqa_dataloaders(cfg)
        cfg['dataset'].pop('split_manifest_file')
        with self.assertRaisesRegex(ValueError,'explicit split manifest'):build_agiqa1k_iqa_dataloaders(cfg)
