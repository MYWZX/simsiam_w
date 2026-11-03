import copy
from pathlib import Path
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]
class OriginalSelectionProjectionLR(unittest.TestCase):
    def test_only_projection_lr_and_identifiers_differ_from_best_baseline(self):
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            with self.subTest(dataset=ds):
                base=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_noce_noneg/{ds}.yaml').read_text())
                actual=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_srocc_projlr01/{ds}.yaml').read_text())
                expected=copy.deepcopy(base)
                for key in ['run_id','output_dir','results_csv']:
                    expected[key]=expected[key].replace('globalpi_ctx5_noce_noneg','globalpi_ctx5_srocc_projlr01_noce_noneg')
                expected['train']['layerwise_lr']['shared_projection_mult']=.1
                self.assertEqual(actual,expected)
                self.assertEqual(actual['train']['selection_metric'],'val_quality_SROCC')
                self.assertEqual(actual['train']['selection_mode'],'max')
                self.assertNotIn('refit_plan',actual['train'])

    def test_runner_has_no_refit_or_pretraining_launch(self):
        script=(ROOT/'scripts/run_ctx5_srocc_projlr01.sh').read_text()
        self.assertNotIn('--refit-plan',script)
        self.assertNotIn('prepare_protocol_refit.py',script)
        self.assertNotIn('train_cm_pretrain.py',script)
