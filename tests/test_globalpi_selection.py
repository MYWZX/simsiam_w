import csv
import json
from pathlib import Path
import tempfile
import unittest
from scripts.compare_globalpi_selection import select_policies

class SelectionPolicyTests(unittest.TestCase):
    def test_different_validation_targets_and_test_independence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'dev.csv'
            def write(test_value):
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                    w.writerow(dict(event='run_start',epoch=0,metrics_json='{}'))
                    for e,(q,p,a,loss) in enumerate([(.9,.7,.2,.3),(.8,.95,.6,.2),(.7,.8,.95,.1)],1):
                        m={'train_processed_steps':10,'val_val_quality_SROCC':q,'val_val_quality_PLCC':p,
                           'val_val_align_SROCC':a,'val_val_align_N':20,'val_val_reg_loss':loss,
                           'test_quality_SROCC':test_value*e}
                        w.writerow(dict(event='epoch',epoch=e,metrics_json=json.dumps(m)))
                    w.writerow(dict(event='final',epoch=3,metrics_json=json.dumps({'best_test':test_value})))
            write(1);a=select_policies({'train':{}},path)
            self.assertEqual([a[k]['epoch'] for k in ['quality_srocc','quality_plcc','joint','loss']],[1,2,3,3])
            self.assertEqual(a['joint']['total_steps'],30)
            write(-100);self.assertEqual(a,select_policies({'train':{}},path))

    def test_joint_without_align_labels_reduces_to_quality(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'dev.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                w.writerow(dict(event='run_start',epoch=0,metrics_json='{}'))
                for e,q in [(1,.9),(2,.8)]:
                    m={'train_processed_steps':5,'val_val_quality_SROCC':q,'val_val_quality_PLCC':q,
                       'val_val_align_N':0,'val_val_reg_loss':1-q}
                    w.writerow(dict(event='epoch',epoch=e,metrics_json=json.dumps(m)))
                w.writerow(dict(event='final',epoch=2,metrics_json='{}'))
            result=select_policies({'train':{}},path)
            self.assertEqual(result['joint']['epoch'],result['quality_srocc']['epoch'])
