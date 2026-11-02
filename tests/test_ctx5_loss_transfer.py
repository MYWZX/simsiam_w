import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import yaml
from scripts.prepare_ctx5_loss_transfer import export_transfer,archive_unstarted

class LossTransferTests(unittest.TestCase):
    def test_numpy_resume_state_is_excluded_and_same_epoch_tensors_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'pretrain';root.mkdir()
            cfg={'train':{'pretrain_selection_metric':'val_loss','pretrain_selection_mode':'min'},
                 'model':{'text_encoder':{'learnable_context_tokens':5}}}
            weight=torch.randn(3,4)
            torch.save({'epoch':7,'cfg':cfg,'model':{'iqa.backbone.fusion.weight':weight},
                        'rng':{'numpy':np.random.get_state()}},root/'best_pretrain.pth')
            with root.with_suffix('.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['event','metrics_json']);w.writeheader()
                w.writerow({'event':'final','metrics_json':json.dumps({'selection_metric':'val_loss','best_epoch':7})})
            dest=export_transfer(root);result=torch.load(dest,weights_only=True)
            self.assertEqual(result['epoch'],7);self.assertNotIn('rng',result)
            torch.testing.assert_close(result['model']['iqa.backbone.fusion.weight'],weight)
            self.assertEqual(export_transfer(root),dest)
            torch.save({'changed':True},root/'best_pretrain.pth')
            with self.assertRaises(ValueError):export_transfer(root)

    def test_archive_only_unstarted_run_and_refuse_training_progress(self):
        previous=os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                output=Path('outputs/run');output.mkdir(parents=True)
                cfg={'output_dir':str(output),'results_csv':'outputs/run.csv','train':{'pretrained_checkpoint':'outputs/pre/best_loss_representation.pth'}}
                cfgpath=Path('cfg.yaml');cfgpath.write_text(yaml.safe_dump(cfg))
                old=yaml.safe_load(yaml.safe_dump(cfg));old['train']['pretrained_checkpoint']='outputs/pre/best_pretrain.pth'
                (output/'config_snapshot.yaml').write_text(yaml.safe_dump(old))
                Path(cfg['results_csv']).write_text('event\nrun_start\nepoch\n')
                with self.assertRaises(ValueError):archive_unstarted(cfgpath)
                self.assertTrue(output.exists())
                Path(cfg['results_csv']).write_text('event\nrun_start\n')
                archive_unstarted(cfgpath)
                self.assertFalse(output.exists())
                self.assertEqual(len(list(Path('outputs/failed_before_training').rglob('run.csv'))),1)
            finally:os.chdir(previous)
