"""Independent subset preserves unaffected membership and fixed-budget refit."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
import yaml
from datasets.multi_iqa_dataset import build_agiqa1k_iqa_dataloaders
from utils.agiqa1k_subset1078 import expected_manifest, file_sha256, RELATIVE_DIR, CSV_NAME
from utils.refit_protocol import split_manifest, make_plan, merge_train_validation
from train_cm_iqa import get_selection_value

ROOT=Path(__file__).resolve().parents[1]

class Subset1078Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);revision=self.root/RELATIVE_DIR;revision.mkdir(parents=True)
        self.csv=revision/CSV_NAME
        frame=pd.DataFrame({'name':[f'{i}.jpg' for i in range(1078)],
            'prompt':['cat']*1078,'answer':['cat']*1078,'MOS':[float(i%11)/2 for i in range(1078)]})
        frame.to_csv(self.csv,index=False)
        self.patcher=patch('utils.agiqa1k_subset1078.SUBSET_CSV_SHA256',file_sha256(self.csv))
        self.patcher.start();self.addCleanup(self.patcher.stop)
        self.manifest=expected_manifest();self.manifest_path=revision/'split_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        image=self.root/'file';image.mkdir()
        for name in frame.name:(image/name).touch()
        self.cfg=yaml.safe_load((ROOT/'configs/agiqa1k_subset1078/both.yaml').read_text())
        self.cfg['dataset']['root_dir']=str(self.root)
        self.cfg['loader'].update(num_workers=0,pin_memory=False)
        ckpt=self.root/'pretrained.pth';ckpt.write_bytes(b'source')
        self.cfg['train']['pretrained_checkpoint']=str(ckpt)

    def test_actual_loaders_membership_and_nonoverlap(self):
        loaders=build_agiqa1k_iqa_dataloaders(self.cfg)
        self.assertEqual([len(l.dataset) for l in loaders],[755,108,215])
        self.assertEqual(len(loaders[0]),94)
        order=torch.randperm(1080,generator=torch.Generator().manual_seed(42)).tolist()
        for loader,indices in zip(loaders,[order[:756],order[756:864],order[864:]]):
            self.assertEqual(list(loader.dataset.indices),[i for i in indices if i<1078])
        manifest=split_manifest(*loaders)
        self.assertEqual(manifest['split_sizes'],{'train':755,'val':108,'test':215})
        self.assertEqual(self.cfg['dataset']['fitted_mos_normalization']['fit_count'],755)

    def test_new_budget_and_combined_training_keep_test(self):
        loaders=build_agiqa1k_iqa_dataloaders(self.cfg)
        source=self.root/'dev.csv'
        with source.open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            for event,e,m in [('run_start',0,{}),('epoch',1,{'val_val_quality_SROCC':.8,'train_processed_steps':94}),('epoch',2,{'val_val_quality_SROCC':.9,'train_processed_steps':94}),('final',2,{})]:
                w.writerow(dict(event=event,epoch=e,metrics_json=json.dumps(m)))
        plan=make_plan(self.cfg,source,loaders,get_selection_value)
        test=list(loaders[2].dataset.indices)
        merged,val=merge_train_validation(*loaders,self.cfg,plan)
        self.assertIsNone(val);self.assertEqual(len(merged.dataset),863)
        self.assertEqual(len(merged),107);self.assertEqual(plan['total_steps'],188)
        self.assertEqual(list(loaders[2].dataset.indices),test)
        self.assertEqual(self.cfg['dataset']['fitted_mos_normalization']['fit_count'],863)
        self.assertTrue(set(merged.dataset.indices).isdisjoint(test))

    def test_missing_manifest_cannot_fall_back_to_random_split(self):
        self.cfg['dataset'].pop('split_manifest_file')
        with self.assertRaisesRegex(ValueError,'explicit split manifest'):
            build_agiqa1k_iqa_dataloaders(self.cfg)

    def test_edited_annotation_is_rejected(self):
        self.csv.write_text(self.csv.read_text().replace('cat','dog',1))
        with self.assertRaisesRegex(ValueError,'CSV changed'):build_agiqa1k_iqa_dataloaders(self.cfg)

    def test_manifest_cannot_silently_reshuffle_test(self):
        self.manifest['split_indices']['train'][0],self.manifest['split_indices']['test'][0]=self.manifest['split_indices']['test'][0],self.manifest['split_indices']['train'][0]
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'manifest differs'):build_agiqa1k_iqa_dataloaders(self.cfg)

    def test_both_configs_share_subset_but_reuse_separate_pretraining(self):
        both=yaml.safe_load((ROOT/'configs/agiqa1k_subset1078/both.yaml').read_text())
        merged=yaml.safe_load((ROOT/'configs/agiqa1k_subset1078/allmerged.yaml').read_text())
        self.assertEqual(both['dataset'],merged['dataset'])
        self.assertNotEqual(both['train']['pretrained_checkpoint'],merged['train']['pretrained_checkpoint'])
        for c in [both,merged]:
            self.assertIn('subset1078',c['run_id']);self.assertFalse(c['model']['quality_text_bank']['enabled'])
            self.assertEqual(c['train']['lambda_neg'],0)

    def test_wrapper_routes_all_and_individual_refit_without_pretraining(self):
        for mode,phase,count in [('all','all',7),('both','refit',2),('allmerged','plan',2)]:
            output=subprocess.check_output(['bash','scripts/run_agiqa1k_subset1078.sh',mode,phase],
                cwd=ROOT,env={**os.environ,'ABLATION_PYTHON':'/bin/echo'},text=True)
            self.assertEqual(len(output.strip().splitlines()),count)
            self.assertNotIn('train_cm_pretrain.py',output)
            if phase=='refit':self.assertIn('--refit-plan outputs/subset1078_plans/both.json',output)

if __name__=='__main__':unittest.main()
