import unittest
from scripts.prepare_protocol_refit import refit_run_name

class RefitNamingTests(unittest.TestCase):
    def test_unmerged_outputs_have_separate_destination(self):
        for ds in ('agiqa1k_subset1078', 'agiqa3k', 'aigciqa2023'):
            name=f'{ds}_zii_noce_noneg_unmerged_seed42'
            self.assertEqual(refit_run_name(name), f'{ds}_zii_noce_noneg_refit80_seed42')

    def test_existing_development_names_unchanged(self):
        self.assertEqual(refit_run_name('agiqa3k_method_dev_seed42'),
                         'agiqa3k_method_refit80_seed42')

    def test_reject_ambiguous_or_already_merged_name(self):
        for name in ('baseline', 'agiqa3k_zii_refit80_seed42'):
            with self.assertRaises(ValueError):
                refit_run_name(name)
