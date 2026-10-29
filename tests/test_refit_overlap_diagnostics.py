import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
import yaml
from utils.refit_protocol import split_manifest, split_overlap_report, ImageSplitOverlapError

class Images(Dataset):
    def __init__(self, root, names):
        self.root = root
        self.df = pd.DataFrame({'name':names, 'MOS':[1.]*len(names), 'prompt':['cat']*len(names)})
    def _path(self,name):return self.root/name
    def __len__(self):return len(self.df)
    def __getitem__(self,index):return index

class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def loaders(self,names):
        ds=Images(self.root,names)
        return tuple(DataLoader(Subset(ds,indices)) for indices in ([0,1],[2],[3]))
    def test_real_overlap_report_contains_indices_labels_and_prompt(self):
        loaders=self.loaders(['a.png','b.png','a.png','d.png'])
        with self.assertRaises(ImageSplitOverlapError) as caught:split_manifest(*loaders)
        report=caught.exception.report
        pair=report['cross_split']['train_val']
        self.assertEqual(pair['image_count'],1)
        self.assertEqual(pair['examples'][0]['train'][0]['row_index'],0)
        self.assertEqual(pair['examples'][0]['val'][0]['row_index'],2)
        self.assertEqual(pair['examples'][0]['val'][0]['prompt'],'cat')
        self.assertEqual(report['cross_split']['train_test']['image_count'],0)
    def test_aliases_resolve_to_same_image_even_with_different_names(self):
        loaders=self.loaders(['a.png','b.png','sub/../a.png','d.png'])
        with self.assertRaises(ImageSplitOverlapError):split_manifest(*loaders)
    def test_same_basename_in_distinct_directories_is_not_collision(self):
        loaders=self.loaders(['one/a.png','b.png','two/a.png','d.png'])
        report=split_overlap_report(loaders)
        self.assertTrue(all(pair['image_count']==0 for pair in report['cross_split'].values()))
        manifest=split_manifest(*loaders)
        self.assertEqual(manifest['split_sizes'],{'train':2,'val':1,'test':1})
    def test_safe_manifest_schema_retains_existing_hash_format(self):
        from utils.refit_protocol import digest
        names=['a','b','c','d'];manifest=split_manifest(*self.loaders(names))
        self.assertEqual(manifest['dataset_ids_sha256'],digest(names))
        self.assertEqual(manifest['test_ids_sha256'],digest(['d']))
    def test_plan_cli_writes_report_but_not_plan_on_failure(self):
        import scripts.prepare_protocol_refit as entry
        prefix=self.root/'plans/run';snapshot=self.root/'snapshot.yaml'
        snapshot.write_text(yaml.safe_dump({'dataset':{'type':'agiqa1k_iqa'}}))
        report=split_overlap_report(self.loaders(['a','b','a','d']))
        with patch('sys.argv',['prepare','--snapshot',str(snapshot),'--csv','unused.csv','--prefix',str(prefix)]),patch.object(entry,'build_dataloaders',return_value=self.loaders(['a','b','a','d'])),patch.object(entry,'make_plan',side_effect=ImageSplitOverlapError(report)),patch('sys.stderr'):
            with self.assertRaises(SystemExit) as caught:entry.main()
        self.assertEqual(caught.exception.code,2)
        path=Path(str(prefix)+'.overlap.json');self.assertTrue(path.exists())
        self.assertEqual(json.loads(path.read_text())['cross_split']['train_val']['image_count'],1)
        self.assertFalse(Path(str(prefix)+'.json').exists())
        self.assertFalse(Path(str(prefix)+'.yaml').exists())

if __name__=='__main__':unittest.main()
