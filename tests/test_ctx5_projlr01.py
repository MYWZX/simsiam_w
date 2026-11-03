import copy
from pathlib import Path
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]
class ProjectionLRConfigs(unittest.TestCase):
    def test_only_projection_lr_and_run_identity_change(self):
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            with self.subTest(dataset=ds):
                base=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_loss_noce_noneg/{ds}.yaml').read_text())
                new=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_loss_projlr01/{ds}.yaml').read_text())
                expected=copy.deepcopy(base)
                for key in ['run_id','output_dir','results_csv']:
                    expected[key]=expected[key].replace('globalpi_ctx5_loss_noce_noneg','globalpi_ctx5_loss_projlr01_noce_noneg')
                expected['train']['layerwise_lr']['shared_projection_mult']=.1
                self.assertEqual(new,expected)
                self.assertEqual(new['train']['pretrained_checkpoint'],base['train']['pretrained_checkpoint'])
