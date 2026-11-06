"""Score-head-only dropout changes with fixed source/splits; no real training."""
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
from scripts import run_score_dropout as runner

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        shutil.copytree(runner.ROOT/'configs/projection_only_v1_baseline',self.root/'configs/projection_only_v1_baseline')
        (self.root/'docs').mkdir();(self.root/'docs/current_baseline.json').write_text('{}');(self.root/'docs/score_dropout.md').write_text('protocol')
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
        scoring={name:[dict(module='2',p=runner.VARIANTS[v])] for name in ('global_quality_head','patch_quality_head','align_global_head','align_patch_head')}
        runner.base.write_new(d/'run_metadata.json',dict(scoring_dropout_initialization=scoring,seed=42,split_seed=c['split_seed'],checkpoint_verified_after_load=True,checkpoint=self.source,
            optimizer_initialization=record,data_split={k:v for k,v in evidence.items() if k!='indices'}))
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','metrics_json']);writer.writeheader()
            for event,m in [('optimizer_initialization',{'groups':record}),('scoring_dropout_initialization',scoring),('final',dict(last_epoch=60,selection_metric='val_quality_SROCC'))]:writer.writerow(dict(event=event,metrics_json=json.dumps(m)))
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in runner.FILES if n!='completion_manifest.json'},csv_sha256=runner.base.digest(table)))
        (d/'unused_weights.pth').write_bytes(b'excluded')
        return d,table

    def test_fixed_splits_config_and_cpu_refusal(self):
        with self.assertRaises(ValueError):runner.config('sdrop010','agiqa1k',3047)
        with self.assertRaises(ValueError):runner.config('sdrop010','agiqa3k',3407)
        for ds in runner.DATASETS:
            original=yaml.safe_load((self.root/f'configs/projection_only_v1_baseline/{ds}.yaml').read_text())
            for v in runner.VARIANTS:
                c=runner.config(v,ds)
                self.assertEqual(c['seed'],42);self.assertEqual(c['split_seed'],3407 if ds=='agiqa1k' else 42)
                expected_model=copy.deepcopy(original['model']);expected_model['heads']['score_dropout']=runner.VARIANTS[v]
                self.assertEqual(c['model'],expected_model)
                self.assertEqual(c['train']['weight_decay'],.002)
                self.assertNotIn('post_load_qa_alpha',c['train'])
                expected=dict(original['train']['layerwise_lr']);expected=dict(original['train']['layerwise_lr'])
                self.assertEqual(c['train']['layerwise_lr'],expected)
        with patch.object(runner,'control',side_effect=ValueError('missing control')):
            with self.assertRaisesRegex(ValueError,'missing control'):runner.run('sdrop010','agiqa1k',dry_run=True)
        with patch.object(runner,'control'),contextlib.redirect_stdout(io.StringIO()):runner.run('sdrop010','agiqa3k',dry_run=True)
        with patch.object(runner,'control'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):runner.run('sdrop010','agiqa3k')

    def test_control_provenance_pack_and_no_overwrite(self):
        controls={}
        for ds in runner.DATASETS:controls[ds]=self.fake('sdrop010',ds,3407 if ds=='agiqa1k' else None)
        # Replace their control labels only in the mocked control resolver; reuse
        # identical actual memberships to exercise package checks.
        for ds in runner.DATASETS:
            for v in ('sdrop010','sdrop030'):
                if v=='sdrop030':self.fake(v,ds,3407 if ds=='agiqa1k' else None)
                runner.verify(v,ds,3407 if ds=='agiqa1k' else None,self.source)
        with patch.object(runner,'control',side_effect=lambda ds,s,f:controls[ds]),contextlib.redirect_stdout(io.StringIO()):
            runner.pack()
            with self.assertRaises(FileExistsError):runner.pack()
            with self.assertRaises(FileExistsError):runner.run('sdrop010','agiqa1k',3407,True)
        with tarfile.open(self.root/f'{runner.VERSION}_1ksplit3407_results.tar.gz') as t:
            names=t.getnames();self.assertIn('source_fingerprint.json',names);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,_=runner.paths('sdrop030','agiqa1k',3407);(d/'console.log').write_text('tampered')
        with self.assertRaises(ValueError):runner.verify('sdrop030','agiqa1k',3407,self.source)


    def test_effective_optimizer_lr_mismatch_rejected(self):
        d,_=self.fake('sdrop010','agiqa3k',None)
        meta=runner.base.read_json(d/'run_metadata.json')
        next(g for g in meta['optimizer_initialization'] if g['name']=='quality_adapter')['lr']=2e-4
        (d/'run_metadata.json').write_text(json.dumps(meta))
        done=runner.base.read_json(d/'completion_manifest.json')
        done['files_sha256']['run_metadata.json']=runner.base.digest(d/'run_metadata.json')
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'optimizer group'):runner.verify('sdrop010','agiqa3k',None,self.source)


    def test_six_run_dispatch_and_no_launch_on_preflight_failure(self):
        with patch.object(runner,'control'),patch.object(runner,'run') as launch:
            runner.run_all()
            self.assertEqual([call.args for call in launch.call_args_list],[(v,ds) for ds in runner.DATASETS for v in runner.VARIANTS])
        with patch.object(runner,'control'),patch.object(runner,'run',side_effect=RuntimeError('stop')) as launch:
            with self.assertRaises(RuntimeError):runner.run_all()
            self.assertEqual(launch.call_count,1)
        runner.paths('sdrop030','aigciqa2023')[0].mkdir(parents=True)
        with patch.object(runner,'control'),patch.object(runner,'run') as launch:
            with self.assertRaises(FileExistsError):runner.run_all()
            launch.assert_not_called()

    def test_actual_dropout_mismatch_rejected(self):
        d,_=self.fake('sdrop010','agiqa3k',None)
        meta=runner.base.read_json(d/'run_metadata.json')
        meta['scoring_dropout_initialization']['global_quality_head'][0]['p']=.2
        (d/'run_metadata.json').write_text(json.dumps(meta))
        done=runner.base.read_json(d/'completion_manifest.json');done['files_sha256']['run_metadata.json']=runner.base.digest(d/'run_metadata.json')
        (d/'completion_manifest.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,'scoring-head dropout'):runner.verify('sdrop010','agiqa3k',None,self.source)

class ActualScoringDropoutTests(unittest.TestCase):
    def test_only_four_heads_change_and_eval_and_weights_match(self):
        from utils.run_metadata import scoring_dropout_initialization
        names=('global_quality_head','patch_quality_head','align_global_head','align_patch_head')
        cfg=projection.config(True);cfg['model'].setdefault('heads',{})['dropout']=.2
        torch.manual_seed(42);old=make(cfg).iqa
        state=torch.random.get_rng_state().clone()
        old_probs={name:m.p for name,m in old.named_modules() if isinstance(m,torch.nn.Dropout)}
        image=torch.randn(3,3,16,16)
        old_out=old.forward_single(image,['cat']*3)
        for probability in (.1,.3):
            c=copy.deepcopy(cfg);c['model']['heads']['score_dropout']=probability
            torch.manual_seed(42);new=make(c).iqa
            torch.testing.assert_close(torch.random.get_rng_state(),state)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            for k,v in new.state_dict().items():torch.testing.assert_close(v,old.state_dict()[k],rtol=0,atol=0)
            new.load_state_dict(old.state_dict(),strict=True)
            for name,module in new.named_modules():
                if isinstance(module,torch.nn.Dropout):
                    expected=probability if any(name.startswith(h+'.') for h in names) else old_probs[name]
                    self.assertEqual(module.p,expected)
            record=scoring_dropout_initialization(new)
            self.assertEqual(set(record),set(names));self.assertEqual(sum(map(len,record.values())),7)
            for head in record.values():self.assertTrue(all(item['p']==probability for item in head))
            out=new.forward_single(image,['cat']*3)
            for key in ('s_quality','s_align'):torch.testing.assert_close(out[key],old_out[key],rtol=0,atol=0)
            optimizer=build_optimizer(c,new)
            for group in optimizer.param_groups:self.assertEqual(group['weight_decay'],cfg['train'].get('weight_decay',.002))

    def test_invalid_dropout_rejected(self):
        for probability in (-.1,1.,float('nan'),float('inf')):
            c=projection.config(True);c['model'].setdefault('heads',{})['score_dropout']=probability
            with self.assertRaises(ValueError):make(c)
