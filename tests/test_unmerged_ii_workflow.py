"""No refit route, unchanged g control, fused-z source, safe completed-run reuse."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import yaml

import scripts.run_unmerged_ii_experiments as entry
from utils.run_metadata import checkpoint_fingerprint

ROOT=Path(__file__).resolve().parents[1]

class ConfigTests(unittest.TestCase):
    def test_z_downstream_only_changes_source_and_run_identifiers(self):
        for ds,path in entry.G_CONFIGS.items():
            original=yaml.safe_load((ROOT/path).read_text())
            new=yaml.safe_load((ROOT/f'configs/zii_noce_noneg/{ds}_unmerged.yaml').read_text())
            expected=copy.deepcopy(original)
            for key in ('run_id','output_dir','results_csv'):expected[key]=new[key]
            expected['train']['pretrained_checkpoint']='outputs/richhf18k_zii_noce_noneg_pretrain_dev_seed42/best_component_anchor_pretrain.pth'
            self.assertEqual(new,expected)
            self.assertEqual(new['train']['selection_metric'],'val_quality_SROCC')
            self.assertEqual(new['train']['lambda_quality_text_ce'],0)
            self.assertEqual(new['train']['lambda_neg'],0)
            self.assertNotIn('refit_plan',new['train'])
    def test_z_pretrain_only_changes_image_input_and_recordkeeping(self):
        original=yaml.safe_load((ROOT/'configs/protocol_v2/pretrain_both_noce_noneg.yaml').read_text())
        new=yaml.safe_load((ROOT/entry.Z_PRETRAIN).read_text())
        for key in ('run_id','output_dir','results_csv'):original[key]=new[key]
        original['model']['image_image_feature']='z'
        original['train'].update(record_step_budget=True,record_pretrain_split_manifest=True)
        self.assertEqual(new,original)
    def test_all_dry_run_contains_no_merge_or_plan_commands(self):
        stream=io.StringIO()
        with patch('sys.argv',['workflow','all','--dry-run']),redirect_stdout(stream):entry.main()
        lines=[l for l in stream.getvalue().splitlines() if l.startswith('RUN OR REUSE:')]
        self.assertEqual(len(lines),7)
        self.assertEqual(sum('train_cm_pretrain.py' in l for l in lines),1)
        for line in lines:
            self.assertNotIn('--refit-plan',line)
            self.assertNotIn('prepare_protocol_refit.py',line)
            self.assertNotIn('refit80',line)

class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.cfg=yaml.safe_load((ROOT/entry.G_CONFIGS['agiqa3k']).read_text())
        self.output=self.root/'completed';self.output.mkdir()
        self.path=self.root/'result.csv';source=self.root/'source.pth';source.write_bytes(b'fixed source')
        self.cfg.update(output_dir=str(self.output),results_csv=str(self.path),run_id='completed')
        self.cfg['train'].update(epochs=2,pretrained_checkpoint=str(source))
        self.snapshot=self.output/'config_snapshot.yaml'
        self.snapshot.write_text(yaml.safe_dump(self.cfg))
        meta={'checkpoint_verified_after_load':True,'checkpoint':checkpoint_fingerprint(source)}
        (self.output/'run_metadata.json').write_text(json.dumps(meta))
        with self.path.open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            for event,e,m in [('run_start',0,{}),('epoch',1,{}),('epoch',2,{}),('final',2,{'best_test':{},'best_val':{}})]:
                w.writerow(dict(event=event,epoch=e,metrics_json=json.dumps(m)))
    def test_valid_finished_run_reused_without_training(self):
        cfg_path=self.root/'cfg.yaml';cfg_path.write_text(yaml.safe_dump(self.cfg))
        with patch.object(entry.subprocess,'run',side_effect=AssertionError('Should not retrain')):
            entry.run_stage(str(cfg_path),'train_cm_iqa.py')
    def test_changed_source_rejected(self):
        Path(self.cfg['train']['pretrained_checkpoint']).write_bytes(b'different')
        with self.assertRaisesRegex(ValueError,'source differs'):entry.verify_completed(self.cfg,'train_cm_iqa.py')
    def test_changed_hyperparameter_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['train']['lambda_srocc_quality']=0
        with self.assertRaisesRegex(ValueError,'config differs'):entry.verify_completed(cfg,'train_cm_iqa.py')
    def test_refit_snapshot_is_never_reused_as_development(self):
        cfg=copy.deepcopy(self.cfg);cfg['train']['refit_plan']='plan.json'
        self.snapshot.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError,'Merged refit'):entry.verify_completed(self.cfg,'train_cm_iqa.py')
    def test_derived_subset_sizes_do_not_force_retraining(self):
        actual=copy.deepcopy(self.cfg);actual['dataset']['actual_split_sizes']={'train':755,'val':108,'test':215}
        actual['dataset']['fitted_mos_normalization']={'mode':'train_minmax','min':0,'max':5}
        self.snapshot.write_text(yaml.safe_dump(actual))
        entry.verify_completed(self.cfg,'train_cm_iqa.py')
    def test_missing_final_never_overwrites_incomplete_result(self):
        lines=self.path.read_text().splitlines();self.path.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaisesRegex(RuntimeError,'incomplete'):entry.verify_completed(self.cfg,'train_cm_iqa.py')

if __name__=='__main__':unittest.main()
