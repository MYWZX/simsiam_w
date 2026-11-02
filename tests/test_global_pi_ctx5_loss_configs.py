import copy
from pathlib import Path
import unittest
import yaml
from train_cm_pretrain import get_pretrain_selection_value
from train_cm_iqa import get_selection_value

ROOT=Path(__file__).resolve().parents[1]

class FiveContextLossConfigs(unittest.TestCase):
    def test_only_selection_and_experiment_identity_change(self):
        for name in ('pretrain','agiqa1k','agiqa3k','aigciqa2023'):
            with self.subTest(name=name):
                base=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_noce_noneg/{name}.yaml').read_text())
                actual=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_loss_noce_noneg/{name}.yaml').read_text())
                expected=copy.deepcopy(base)
                for key in ('run_id','output_dir','results_csv'):
                    expected[key]=expected[key].replace('globalpi_ctx5_','globalpi_ctx5_loss_')
                if name=='pretrain':
                    expected['train'].update(pretrain_selection_metric='val_loss',pretrain_selection_mode='min')
                else:
                    expected['train'].update(selection_metric='val_reg_loss',selection_mode='min')
                    expected['train']['pretrained_checkpoint']=expected['train']['pretrained_checkpoint'].replace('globalpi_ctx5_','globalpi_ctx5_loss_').replace('best_component_anchor_pretrain.pth','best_loss_representation.pth')
                self.assertEqual(actual,expected)
                fn=get_pretrain_selection_value if name=='pretrain' else get_selection_value
                metrics={'val_loss':.2,'val_reg_loss':.3,'val_quality_SROCC':.9,'val_component_anchor_SROCC':.95}
                value,maximize,key=fn(metrics,actual)
                self.assertFalse(maximize)
                self.assertEqual(key,'val_loss' if name=='pretrain' else 'val_reg_loss')
                self.assertEqual(value,.2 if name=='pretrain' else .3)
