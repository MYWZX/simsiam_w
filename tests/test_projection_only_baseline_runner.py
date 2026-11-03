"""Runner tests use tiny fake artifacts and never execute a trainer."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import yaml
from scripts import run_projection_only_v1_baseline as runner


class IndependentBaselineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('global_pi_ctx5_noce_noneg','projection_only_v1_baseline'):
            shutil.copytree(runner.ROOT/'configs'/name,self.root/'configs'/name)
        self.patch=patch.multiple(runner,ROOT=self.root,CONFIG=self.root/'configs/projection_only_v1_baseline')
        self.patch.start();self.addCleanup(self.patch.stop)

    def fake_completed(self,stage):
        cfg=runner.validate(stage);d,table=runner.paths(stage);d.mkdir(parents=True)
        source=None if stage=='pretrain' else runner.completed('pretrain')
        runner.write_new(d/'launch_manifest.json',dict(version=runner.VERSION,stage=stage,
            config_sha256=runner.digest(runner.CONFIG/f'{stage}.yaml'),source=source))
        (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
        (d/'console.log').write_text('fake process log\n')
        metric='val_component_anchor_SROCC' if stage=='pretrain' else 'val_quality_SROCC'
        final={'selection_metric':metric}
        files=['config_snapshot.yaml','console.log','run_metadata.json']
        if stage=='pretrain':
            checkpoint=d/'best_component_anchor_pretrain.pth'
            import torch
            torch.save(dict(cfg=cfg,run_id=cfg['run_id'],epoch=5,checkpoint_scope='representation',
                selection_metric=metric,model={k:torch.zeros(1) for k in (
                    'iqa.backbone.fusion.semantic_proj.weight','iqa.quality_adapter.weight',
                    'iqa.shared_image_projector.weight','iqa.shared_text_projector.weight',
                    'iqa.local_patch_proj.weight','iqa.quality_local_encoder.weight')}),checkpoint)
            final.update(best_component_anchor_epoch=5,best_component_anchor_checkpoint=str(checkpoint))
            runtime={}
        else:
            runner.write_new(d/'source_fingerprint.json',source);files.append('source_fingerprint.json')
            runtime={'checkpoint_verified_after_load':True,'checkpoint':{'sha256':source['sha256']}}
        runner.write_new(d/'run_metadata.json',runtime)
        with table.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','metrics_json']);w.writeheader();w.writerow(dict(event='final',metrics_json=json.dumps(final)))
        done={'files_sha256':{p:runner.digest(d/p) for p in files},'csv_sha256':runner.digest(table),
              'launch_sha256':runner.digest(d/'launch_manifest.json')}
        if stage=='pretrain':done['checkpoint_sha256']=runner.digest(checkpoint)
        runner.write_new(d/'completion_manifest.json',done)
        return d

    def test_configs_exact_baseline_and_dry_run_no_trainer(self):
        for stage in runner.STAGES:runner.validate(stage)
        with patch.object(runner.subprocess,'Popen',side_effect=AssertionError('trainer forbidden')):
            with contextlib.redirect_stdout(io.StringIO()):runner.run('pretrain',True)
        self.assertFalse(runner.paths('pretrain')[0].exists())
        with self.assertRaises(FileNotFoundError):runner.run('agiqa3k',True)

    def test_source_identity_pack_and_tamper(self):
        d=self.fake_completed('pretrain');fingerprint=runner.completed('pretrain')
        self.assertEqual(fingerprint['selected_epoch'],5)
        for stage in runner.STAGES[1:]:self.fake_completed(stage);runner.completed(stage)
        with contextlib.redirect_stdout(io.StringIO()):runner.pack('downstream');runner.pack('pretrain')
        import tarfile
        with tarfile.open(self.root/f'{runner.VERSION}_downstream_results.tar.gz') as t:
            names=t.getnames();self.assertTrue(any(n.endswith('console.log') for n in names))
            self.assertIn('source_fingerprint.json',names)
            self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
        with self.assertRaises(FileExistsError):runner.pack('pretrain')
        with self.assertRaises(FileExistsError):runner.run('pretrain',True)
        (d/'best_component_anchor_pretrain.pth').write_bytes(b'changed')
        with self.assertRaises(ValueError):runner.run('agiqa3k',True)

    def test_wrong_mode_config_and_incomplete_run_rejected(self):
        d=self.fake_completed('pretrain')
        (d/'completion_manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):runner.completed('pretrain')
        p=runner.CONFIG/'agiqa1k.yaml';c=yaml.safe_load(p.read_text());c['train']['layerwise_lr']['shared_projection_mult']=.1
        p.write_text(yaml.safe_dump(c))
        with self.assertRaises(ValueError):runner.validate('agiqa1k')

    def test_cpu_refuses_before_creating_outputs(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):runner.run('pretrain')
        self.assertFalse(runner.paths('pretrain')[0].exists())

    def test_launch_only_one_stage_with_logs_and_verified_completion(self):
        commands=[]
        class FakeTrainer:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg=runner.configuration('pretrain');d,table=runner.paths('pretrain')
                (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                import torch
                checkpoint=d/'best_component_anchor_pretrain.pth'
                state={k:torch.zeros(1) for k in ('iqa.backbone.fusion.semantic_proj.weight',
                    'iqa.quality_adapter.weight','iqa.shared_image_projector.weight',
                    'iqa.shared_text_projector.weight','iqa.local_patch_proj.weight','iqa.quality_local_encoder.weight')}
                torch.save(dict(cfg=cfg,run_id=cfg['run_id'],epoch=5,checkpoint_scope='representation',
                    selection_metric='val_component_anchor_SROCC',model=state),checkpoint)
                with table.open('w') as f:
                    w=csv.DictWriter(f,fieldnames=['event','metrics_json']);w.writeheader()
                    w.writerow(dict(event='final',metrics_json=json.dumps(dict(selection_metric='val_component_anchor_SROCC',
                        best_component_anchor_epoch=5,best_component_anchor_checkpoint=str(checkpoint)))))
                self.stdout=iter(['fake pretrain log\n'])
            def wait(self):return 0
        with patch('torch.cuda.is_available',return_value=True),patch.object(runner.subprocess,'Popen',FakeTrainer),patch('utils.run_metadata.collect_run_metadata',return_value={'test':True}):
            with contextlib.redirect_stdout(io.StringIO()):runner.run('pretrain')
        self.assertEqual(len(commands),1)
        self.assertEqual(commands[0][1],'train_cm_pretrain.py')
        self.assertNotIn('--resume',commands[0])
        self.assertEqual(runner.completed('pretrain')['selected_epoch'],5)
        self.assertEqual((runner.paths('pretrain')[0]/'console.log').read_text(),'fake pretrain log\n')
        self.assertFalse(runner.paths('agiqa1k')[0].exists())

    def test_checkpoint_metadata_rejects_other_mode_and_selection(self):
        import torch
        d=self.fake_completed('pretrain');p=d/'best_component_anchor_pretrain.pth'
        original=torch.load(p,weights_only=False)
        for field,value in [('run_id','old_experiment'),('selection_metric','val_loss'),('epoch',14)]:
            c=dict(original);c[field]=value;torch.save(c,p)
            with self.assertRaises(ValueError):runner.verify_checkpoint(p,runner.configuration('pretrain'),5)
        c=dict(original);c['cfg']=json.loads(json.dumps(original['cfg']))
        c['cfg']['model']['fusion']['mode']='attention';torch.save(c,p)
        with self.assertRaises(ValueError):runner.verify_checkpoint(p,runner.configuration('pretrain'),5)
        c=dict(original);c['model']=dict(original['model']);c['model']['iqa.backbone.fusion.attn.in_proj_weight']=torch.zeros(1)
        torch.save(c,p)
        with self.assertRaises(ValueError):runner.verify_checkpoint(p,runner.configuration('pretrain'),5)
