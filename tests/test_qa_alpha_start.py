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
from scripts import run_qa_alpha_start as runner


class AlphaOverrideTests(unittest.TestCase):
    def test_complete_transfer_then_only_scalar_changes_and_keeps_gradient(self):
        cfg=projection.config(True);source=make(cfg)
        with torch.no_grad():source.iqa.quality_adapter_gate_logit.fill_(torch.logit(torch.tensor(.12)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.pth';torch.save({'model':_extract_representation_state(source.state_dict())},p)
            for alpha in (.05,.2):
                model=make(cfg).iqa;load_pretrained_backbone(model,p,load_scope='representation')
                before={k:v.clone() for k,v in model.state_dict().items()}
                param=model.quality_adapter_gate_logit
                config=copy.deepcopy(cfg);config['train']['post_load_qa_alpha']=alpha
                record=apply_post_load_qa_alpha(model,config,p)
                self.assertAlmostEqual(record['loaded_alpha'],.12,places=6)
                self.assertAlmostEqual(record['applied_alpha'],alpha,places=6)
                self.assertIs(param,model.quality_adapter_gate_logit)
                for k,v in model.state_dict().items():
                    if k!='quality_adapter_gate_logit':torch.testing.assert_close(v,before[k],rtol=0,atol=0)
                out=model.forward_single(torch.randn(3,3,16,16),['cat']*3)
                grad=torch.autograd.grad(out['s_quality'].sum(),param)[0]
                self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().item(),0)
                config['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'fusion_mult':.1,'shared_projection_mult':1.})
                optimizer=build_optimizer(config,model)
                lrs={g['name']:g['lr'] for g in optimizer.param_groups}
                self.assertEqual(lrs['fusion'],1e-5);self.assertEqual(lrs['shared_projection'],1e-4)

    def test_no_override_is_noop_and_invalid_inputs_rejected(self):
        cfg=projection.config(True);model=make(cfg).iqa
        before=model.quality_adapter_gate_logit.clone()
        self.assertIsNone(apply_post_load_qa_alpha(model,cfg,''))
        torch.testing.assert_close(before,model.quality_adapter_gate_logit)
        for value in [0,1,-1,float('nan'),float('inf')]:
            c=copy.deepcopy(cfg);c['train']['post_load_qa_alpha']=value
            with self.assertRaises(ValueError):apply_post_load_qa_alpha(model,c,'source')
        cfg['train']['post_load_qa_alpha']=.05
        with self.assertRaises(ValueError):apply_post_load_qa_alpha(model,cfg,'')
        model.quality_adapter_gate_logit.requires_grad_(False)
        with self.assertRaises(ValueError):apply_post_load_qa_alpha(model,cfg,'source')


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
            config_sha256=runner.base.digest(self.root/f'configs/qa_alpha_start/{v}/{ds}.yaml'),source=self.pin))
        (d/'config_snapshot.yaml').write_text(yaml.safe_dump(c));(d/'console.log').write_text('fake log')
        alpha=dict(requested=runner.VARIANTS[v],loaded_alpha=.12,applied_alpha=runner.VARIANTS[v],trainable=True)
        runner.base.write_new(d/'run_metadata.json',dict(checkpoint_verified_after_load=True,
            checkpoint={'sha256':self.pin['sha256']},qa_alpha_initialization=alpha))
        with table.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','metrics_json']);w.writeheader()
            for event,m in [('post_load_qa_alpha',alpha),('final',dict(selection_metric='val_quality_SROCC',last_epoch=60))]:w.writerow(dict(event=event,metrics_json=json.dumps(m)))
        names=('config_snapshot.yaml','run_metadata.json','launch_manifest.json','source_fingerprint.json','console.log')
        runner.base.write_new(d/'completion_manifest.json',dict(files_sha256={n:runner.base.digest(d/n) for n in names},csv_sha256=runner.base.digest(table)))
        return d

    def test_six_configs_pinned_source_and_no_train_dryrun(self):
        for v in runner.VARIANTS:
            for ds in runner.DATASETS:
                cfg=runner.config(v,ds)
                self.assertEqual(cfg['train']['layerwise_lr']['shared_projection_mult'],1.)
                with contextlib.redirect_stdout(io.StringIO()):runner.run(v,ds,True)
                self.assertFalse(runner.paths(v,ds)[0].exists())
        with patch.object(runner.base,'completed',return_value={**self.pin,'sha256':'wrong'}):
            with self.assertRaises(ValueError):runner.source()
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):runner.run('alpha005','agiqa3k')
        self.assertFalse(runner.paths('alpha005','agiqa3k')[0].exists())

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
        with self.assertRaises(FileExistsError):runner.run('alpha005','agiqa3k',True)
        d=runner.paths('alpha005','agiqa3k')[0];(d/'console.log').write_text('changed')
        with self.assertRaises(ValueError):runner.verify('alpha005','agiqa3k',self.pin)

    def test_other_hyperparameter_change_rejected(self):
        p=self.root/'configs/qa_alpha_start/alpha005/agiqa1k.yaml'
        c=yaml.safe_load(p.read_text());c['train']['lr']=.01;p.write_text(yaml.safe_dump(c))
        with self.assertRaises(ValueError):runner.config('alpha005','agiqa1k')
