import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Subset
from utils.clip_experiments import configure_clip_large, training_quality_centers, calibrate_quality_centers


class LabelDataset:
    quality = np.array([0., 20., 40., 60., 80., 100.])
    mos_max = 100.

    def __len__(self):
        return len(self.quality)

    def __getitem__(self, i):
        raise AssertionError('Calibration must not load images or apply augmentation')


class ClipExperimentTests(unittest.TestCase):
    def test_nested_training_subset_excludes_held_out_extremes(self):
        train = Subset(Subset(LabelDataset(), [5, 1, 2, 3, 4, 0]), [1, 2, 3, 4])
        centers, count = training_quality_centers(train)
        np.testing.assert_allclose(centers, [.23, .35, .5, .65, .77])
        self.assertEqual(count, 4)

    def test_calibration_changes_only_centers_and_records_source(self):
        bank = SimpleNamespace(centers=torch.tensor([.1, .2, .3, .4, .5]),
                               class_context=torch.randn(5, 4, 8))
        context = bank.class_context.clone()
        cfg = {'model': {'quality_text_bank': {}}}
        details = calibrate_quality_centers(SimpleNamespace(quality_text_bank=bank),
                                          Subset(LabelDataset(), [1, 2, 3, 4]), cfg)
        self.assertTrue(torch.equal(bank.class_context, context))
        self.assertEqual(details['source'], 'training_labels_only')
        np.testing.assert_allclose(bank.centers, cfg['model']['quality_text_bank']['centers'])

    def test_large_preserves_local_and_fusion_configuration(self):
        cfg = {'model': {'fusion': {'embed_dim': 256}, 'local_encoder': {'output_dim': 256}}}
        configure_clip_large(cfg)
        self.assertEqual(cfg['model']['global_encoder']['output_dim'], 768)
        self.assertEqual(cfg['model']['text_encoder']['clip_model_name'], 'ViT-L-14')
        self.assertEqual(cfg['model']['fusion']['embed_dim'], 256)


if __name__ == '__main__':
    unittest.main()
