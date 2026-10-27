"""Prevent silently misread labels and invalid metric reports."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import savemat
from datasets.multi_iqa_dataset import _load_mat_vector, _load_prompts_from_file, AIGCIQA2023IQADataset
from train_cm_iqa import compute_regression_metrics, require_validation_data

class AuditDataIntegrityTests(unittest.TestCase):
    def test_mat_explicit_key_never_falls_back_and_implicit_requires_unique_vector(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scores.mat';savemat(p,{'MOSz':np.array([1.,2.,3.])})
            np.testing.assert_array_equal(_load_mat_vector(p,'MOSz'),[1.,2.,3.])
            np.testing.assert_array_equal(_load_mat_vector(p,None),[1.,2.,3.])
            with self.assertRaises(KeyError):_load_mat_vector(p,'wrong_key')
            savemat(p,{'first':np.array([1.,2.]),'second':np.array([3.,4.])})
            with self.assertRaises(RuntimeError):_load_mat_vector(p,None)
            for values in [np.array([np.nan,1.]),np.ones((2,2)),np.empty((0,))]:
                savemat(p,{'MOSz':values})
                with self.assertRaises(ValueError):_load_mat_vector(p,'MOSz')

    def test_2023_cannot_silently_truncate_images_or_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'images').mkdir()
            for i in range(3):Image.new('RGB',(16,16)).save(root/'images'/f'{i}.png')
            savemat(root/'q.mat',{'MOSz':np.array([20.,40.])})
            savemat(root/'a.mat',{'MOSz':np.array([30.,50.,60.])})
            kwargs=dict(root_dir=d,image_dir='images',mosz1_file='q.mat',mosz3_file='a.mat',mosz1_key='MOSz',mosz3_key='MOSz',mos_max=100.,use_perturb=False)
            with self.assertRaisesRegex(ValueError,'count mismatch'):AIGCIQA2023IQADataset(**kwargs)
            savemat(root/'q.mat',{'MOSz':np.array([20.,40.,80.])})
            ds=AIGCIQA2023IQADataset(**kwargs)
            self.assertEqual(len(ds),3);self.assertAlmostEqual(ds[2]['quality_score'].item(),.8,places=6)
            with self.assertRaises(FileNotFoundError):_load_prompts_from_file(root/'missing.csv')

    def test_data_manifest_is_reproducible_and_detects_cross_split_image_duplicates(self):
        from scripts.audit_downstream_data import audit
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'images').mkdir();rows=[]
            for i in range(20):
                Image.new('RGB',(16,16)).save(root/'images'/f'{i}.png')
                rows.append(dict(name=f'{i}.png',prompt=f'prompt {i//2}',MOS=1.+i*.1))
            pd.DataFrame(rows).to_csv(root/'data.csv',index=False)
            c={'seed':42,'dataset':{'type':'agiqa1k_iqa','root_dir':d,'image_dir':'images','csv_file':'data.csv','use_perturb':False},'loader':{'batch_size':2,'num_workers':0}}
            a=audit(c);b=audit(c)
            self.assertEqual(a,b);self.assertEqual(a['splits']['test']['n'],4)
            for row in rows:row['name']='0.png'
            pd.DataFrame(rows).to_csv(root/'data.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Data leakage'):audit(c)

    def test_empty_validation_cannot_be_silently_used_for_epoch_selection(self):
        from torch.utils.data import DataLoader
        for loader in [None,[],DataLoader([1],batch_size=1)]:
            with self.assertRaisesRegex(ValueError,'fixed-budget refit'):require_validation_data(loader)
        require_validation_data(DataLoader([1,2],batch_size=2))

    def test_metrics_reject_nonfinite_and_mismatched_data_but_keep_empty_align(self):
        for target,pred in [([1.,2.],[1.]),([1.,np.nan],[1.,2.]),([1.,2.],[np.inf,2.])]:
            with self.assertRaises(ValueError):compute_regression_metrics(target,pred)
        self.assertEqual(compute_regression_metrics([],[])['N'],0)
        m=compute_regression_metrics([.2],[.5]);self.assertAlmostEqual(m['MAE'],.3);self.assertAlmostEqual(m['RMSE'],.3)
        m=compute_regression_metrics([.1,.2,.3],[.2,.4,.6]);self.assertAlmostEqual(m['PLCC'],1.)

if __name__=='__main__':unittest.main()
