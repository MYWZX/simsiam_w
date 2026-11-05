"""Real small-model beta loading/gradient checks and fake runner artifacts."""
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
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import build_optimizer
from utils.quality_mix_tuning import record_quality_mix_start
from utils.refit_protocol import digest
from scripts import run_beta_start as runner


class ActualBetaTests(unittest.TestCase):
    def test_start_survives_representation_load_and_remains_trainable(self):
        source=make(projection.config(True))
        with torch.no_grad():source.iqa.quality_mix_logit.fill_(4.)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.pth'
            state=_extract_representation_state(source.state_dict())
            state['iqa.quality_mix_logit']=source.iqa.quality_mix_logit.detach().clone()
            torch.save({'model':state},path)
            for beta in (.25,.75):
                cfg=projection.config(True)
                cfg['model'].setdefault('nss_text_gate',{})['quality_mix_init']=math.log(beta/(1-beta))
                cfg['train']['record_quality_mix_initialization']=True
                model=make(cfg).iqa
                load_pretrained_backbone(model,path,load_scope='representation')
                for k,v in model.state_dict().items():
                    if 'iqa.'+k in state and k!='quality_mix_logit':torch.testing.assert_close(v,state['iqa.'+k],rtol=0,atol=0)
                before={k:v.clone() for k,v in model.state_dict().items()}
                record=record_quality_mix_start(model,cfg)
                self.assertAlmostEqual(record['applied_beta'],beta,places=6)
                for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
                out=model.forward_single(torch.randn(3,3,16,16),['cat']*3)
                grad=torch.autograd.grad(out['s_quality'].sum(),model.quality_mix_logit)[0]
                self.assertTrue(torch.isfinite(grad));self.assertGreater(grad.abs().item(),0)
                cfg['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'fusion_mult':.1,'shared_projection_mult':1.})
                opt=build_optimizer(cfg,model)
                group=next(g for g in opt.param_groups if any(p is model.quality_mix_logit for p in g['params']))
                self.assertEqual(group['name'],'quality');self.assertEqual(group['lr'],1e-4)
                with torch.no_grad():model.quality_mix_logit.fill_(0.)
                with self.assertRaises(ValueError):record_quality_mix_start(model,cfg)

    def test_old_configs_noop_and_unsupported_mode_refused(self):
        cfg=projection.config(True);model=make(cfg).iqa
        self.assertIsNone(record_quality_mix_start(model,cfg))
        cfg['train']['record_quality_mix_initialization']=True
        model.quality_global_only=True
        with self.assertRaises(ValueError):record_quality_mix_start(model,cfg)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        shutil.copytree(runner.ROOT/'configs/projection_only_v1_baseline',self.root/'configs/projection_only_v1_baseline')
        (self.root/'docs').mkdir();(self.root/'docs/current_baseline.json').write_text('{}');(self.root/'docs/beta_start.md').write_text('protocol')
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
        record=dict(config_logit=c['model']['nss_text_gate']['quality_mix_init'],applied_beta=runner.VARIANTS[v],trainable=True)
        runner.base.write_new(d/'run_metadata.json',dict(seed=42,split_seed=c['split_seed'],checkpoint_verified_after_load=True,checkpoint=self.source,
            quality_mix_initialization=record,data_split={k:v for k,v in evidence.items() if k!='indices'}))
        with table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['event','metrics_json']);writer.writeheader()
            for event,m in [('quality_mix_initialization',record),('final',dict(last_epoch=60,selection_metric='val_quality_SROCC'))]:writer.writerow(dict(event=event,metrics_json=json.dumps(m)))
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in runner.FILES if n!='completion_manifest.json'},csv_sha256=runner.base.digest(table)))
        (d/'unused_weights.pth').write_bytes(b'excluded')
        return d,table

    def test_explicit_split_and_single_parameter_config(self):
        with self.assertRaises(ValueError):runner.config('beta025','agiqa1k')
        with self.assertRaises(ValueError):runner.config('beta025','agiqa3k',3407)
        with self.assertRaises(ValueError):runner.config('control','agiqa1k',3407)
        for split in (3047,3407):
            for v in ('beta025','beta075'):
                c=runner.config(v,'agiqa1k',split)
                self.assertEqual(c['seed'],42);self.assertEqual(c['split_seed'],split)
                self.assertEqual(c['train']['weight_decay'],.002)
                self.assertNotIn('post_load_qa_alpha',c['train'])
                self.assertAlmostEqual(torch.sigmoid(torch.tensor(c['model']['nss_text_gate']['quality_mix_init'])).item(),runner.VARIANTS[v],places=6)
        with patch.object(runner,'control',side_effect=ValueError('missing control')):
            with self.assertRaisesRegex(ValueError,'missing control'):runner.run('beta025','agiqa1k',3047,True)
        with patch.object(runner,'control'),contextlib.redirect_stdout(io.StringIO()):runner.run('beta025','agiqa3k',dry_run=True)
        with patch.object(runner,'control'),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):runner.run('beta025','agiqa3k')

    def test_control_provenance_pack_and_no_overwrite(self):
        controls={}
        for ds in runner.DATASETS:controls[ds]=self.fake('beta025',ds,3407 if ds=='agiqa1k' else None)
        # Replace their control labels only in the mocked control resolver; reuse
        # identical actual memberships to exercise package checks.
        for ds in runner.DATASETS:
            for v in ('beta025','beta075'):
                if v=='beta075':self.fake(v,ds,3407 if ds=='agiqa1k' else None)
                runner.verify(v,ds,3407 if ds=='agiqa1k' else None,self.source)
        with patch.object(runner,'control',side_effect=lambda ds,s,f:controls[ds]),contextlib.redirect_stdout(io.StringIO()):
            runner.pack(3407)
            with self.assertRaises(FileExistsError):runner.pack(3407)
            with self.assertRaises(FileExistsError):runner.run('beta025','agiqa1k',3407,True)
        with tarfile.open(self.root/f'{runner.VERSION}_1ksplit3407_results.tar.gz') as t:
            names=t.getnames();self.assertIn('source_fingerprint.json',names);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,_=runner.paths('beta075','agiqa1k',3407);(d/'console.log').write_text('tampered')
        with self.assertRaises(ValueError):runner.verify('beta075','agiqa1k',3407,self.source)

    def test_3047_control_is_not_automatically_launched(self):
        with self.assertRaises(FileNotFoundError):runner.control('agiqa1k',3047,self.source)
        self.fake('control','agiqa1k',3047)
        runner.control('agiqa1k',3047,self.source)
