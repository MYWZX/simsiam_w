import copy
from pathlib import Path
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]

class ContextFiveConfigs(unittest.TestCase):
    def test_only_prompt_context_count_source_and_run_names_change(self):
        for name in ('pretrain', 'agiqa1k', 'agiqa3k', 'aigciqa2023'):
            with self.subTest(name=name):
                base=yaml.safe_load((ROOT/f'configs/global_pi_noce_noneg/{name}.yaml').read_text())
                actual=yaml.safe_load((ROOT/f'configs/global_pi_ctx5_noce_noneg/{name}.yaml').read_text())
                expected=copy.deepcopy(base)
                expected['model']['text_encoder']['learnable_context_tokens']=5
                for key in ('run_id','output_dir','results_csv'):
                    expected[key]=expected[key].replace('globalpi_','globalpi_ctx5_')
                if name!='pretrain':
                    expected['train']['pretrained_checkpoint']=expected['train']['pretrained_checkpoint'].replace('globalpi_','globalpi_ctx5_')
                self.assertEqual(actual,expected)
                self.assertFalse(actual['model']['quality_text_bank']['enabled'])

    def test_runner_is_separate_and_preserves_original_stages(self):
        base=(ROOT/'scripts/run_global_pi_experiment.sh').read_text()
        expected=base.replace('configs/global_pi_noce_noneg','configs/global_pi_ctx5_noce_noneg').replace('globalpi_','globalpi_ctx5_')
        self.assertEqual((ROOT/'scripts/run_global_pi_ctx5_experiment.sh').read_text(),expected)
