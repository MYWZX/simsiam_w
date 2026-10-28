import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import yaml

from datasets.mixed_pretrain_dataset import SubsetWithWeights
from utils.pretrain_refit import dataset_manifest, merged_richhf_loader, pretrain_budget, run_pretrain_refit

ROOT = Path(__file__).resolve().parents[1]

class Raw(Dataset):
    def __init__(self, root, split, train=False):
        self.root = root
        self.train = train
        self.df = pd.DataFrame({'name': [f'{split}{i}.png' for i in range(4)]})
        self.csv_paths = [root/f'{split}.csv']
        if not self.csv_paths[0].exists(): self.df.to_csv(self.csv_paths[0], index=False)
        for name in self.df.name: (root/name).touch(exist_ok=True)
    def _find_path(self, row): return self.root/row['name']
    def __len__(self): return len(self.df)
    def __getitem__(self, i): return torch.tensor([float(i)])

class MergedPretrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = yaml.safe_load((ROOT/'configs/allmerged_noce_noneg/pretrain_dev.yaml').read_text())
        self.cfg['loader'].update(batch_size=2, num_workers=0, pin_memory=False)
        self.ds = [Raw(self.root, split, split=='train') for split in ['train','val','test']]
        wrapped = SubsetWithWeights(self.ds[0], list(range(4)), 1.)
        original = ConcatDataset([wrapped])
        sampler = WeightedRandomSampler(torch.ones(4), 4, replacement=True)
        self.loaders = (DataLoader(original, batch_size=2, sampler=sampler, drop_last=True),
            DataLoader(ConcatDataset([self.ds[1]]), batch_size=2),
            DataLoader(ConcatDataset([self.ds[2]]), batch_size=2))
    def build(self, cfg, train):
        self.assertTrue(train); self.assertEqual(cfg['split'],'val')
        self.assertEqual(cfg['supervision_view'],'three_view')
        return Raw(self.root, 'val', train)
    def test_merge_membership_training_transform_and_test_untouched(self):
        before = dataset_manifest(self.ds)
        loader, manifest = merged_richhf_loader(self.cfg, self.loaders, self.build)
        self.assertEqual(manifest, before)
        self.assertEqual(len(loader.dataset), 8)
        self.assertTrue(loader.dataset.datasets[1].train)
        self.assertEqual(len(loader.dataset.datasets[0]), 4)
        self.assertEqual(manifest['counts'], {'train':4,'val':4,'test':4})
        self.assertEqual(dataset_manifest(self.ds)['test_ids_sha256'],before['test_ids_sha256'])
    def test_overlap_rejected(self):
        self.ds[1].df.loc[0,'name']='train0.png'
        with self.assertRaisesRegex(ValueError,'overlap'):dataset_manifest(self.ds)
    def test_label_csv_fingerprint_changes(self):
        before=dataset_manifest(self.ds)
        self.ds[1].csv_paths[0].write_text('changed labels')
        self.assertNotEqual(before,dataset_manifest(self.ds))
    def test_validation_budget_ignores_test_and_requires_recorded_steps(self):
        path=self.root/'dev.csv'
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            for event,e,m in [('run_start',0,{}),('epoch',1,{'val_component_anchor_SROCC':.8,'train_processed_steps':2,'test_score':0}),('epoch',2,{'val_component_anchor_SROCC':.6,'train_processed_steps':2,'test_score':1}),('final',2,{})]:
                w.writerow(dict(event=event,epoch=e,metrics_json=json.dumps(m)))
        b=pretrain_budget(path,self.cfg,2)
        self.assertEqual(b['epoch'],1);self.assertEqual(b['total_steps'],2)
        with self.assertRaises(ValueError):pretrain_budget(path,self.cfg,3)
    def test_exact_partial_epoch_budget_no_eval(self):
        loader,_=merged_richhf_loader(self.cfg,self.loaders,self.build)
        model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.1)
        seen=[];logs=[]
        def train(model,loader,opt,scaler,device,cfg,epoch,optimizer_step_callback):
            for sample in loader:
                opt.zero_grad();model(sample).sum().backward();opt.step()
                optimizer_step_callback(True);seen.append(sample.shape[0])
            return {}
        stats=run_pretrain_refit(model=model,loader=loader,optimizer=opt,device=torch.device('cpu'),
            cfg=self.cfg,budget={'total_steps':5},original_batches=2,
            logger=SimpleNamespace(log=lambda **kw:logs.append(kw)),train_epoch=train)
        self.assertEqual(seen,[2]*5);self.assertEqual(stats['training_steps'],5)
        self.assertEqual(stats['optimizer_updates'],5);self.assertEqual(stats['epoch'],2)
        self.assertEqual([r['event'] for r in logs],['refit_epoch','refit_epoch'])
    def test_new_downstream_source_and_original_configs_preserved(self):
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            cfg=yaml.safe_load((ROOT/f'configs/allmerged_noce_noneg/{ds}.yaml').read_text())
            self.assertTrue(cfg['train']['pretrained_checkpoint'].endswith('/final_merged_pretrain.pth'))
            self.assertIn('_allmerged_',cfg['run_id'])
            old=yaml.safe_load((ROOT/f'configs/protocol_v2/{ds}_both_noce_noneg.yaml').read_text())
            self.assertIn('best_component_anchor_pretrain.pth',old['train']['pretrained_checkpoint'])
            for key in ('model','dataset','loader'):self.assertEqual(cfg[key],old[key])
    def test_unsupported_sampling_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['train']['quality_balanced_sampling']['enabled']=True
        with self.assertRaises(ValueError):merged_richhf_loader(cfg,self.loaders,self.build)

    def test_real_pretrain_step_records_callback_and_successful_update(self):
        import contextlib
        import io
        from test_dual_quality_residual import residual_config
        from test_dual_quality_representation import batch
        from test_projection_ablations import make
        from train_cm_pretrain import train_one_epoch, build_optimizer
        from torch.amp import GradScaler
        cfg=residual_config();cfg['train']['record_step_budget']=True
        model=make(cfg);optimizer=build_optimizer(cfg,model);calls=[]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics=train_one_epoch(model,[batch()],optimizer,GradScaler('cpu',enabled=False),
                torch.device('cpu'),cfg,1,optimizer_step_callback=calls.append)
        self.assertEqual(calls,[True]);self.assertEqual(metrics['processed_steps'],1)
        self.assertEqual(metrics['optimizer_updates'],1)

    def test_refit_entrypoint_publishes_fixed_end_representation(self):
        from unittest.mock import patch
        from contextlib import ExitStack, redirect_stdout
        import io
        import scripts.refit_merged_pretrain as runner
        self.cfg['dataset']['pretrain_development_manifest']=dataset_manifest(self.ds)
        snapshot=self.root/'snapshot.yaml';snapshot.write_text(yaml.safe_dump(self.cfg))
        source=self.root/'dev.csv'
        with source.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            for event,e,m in [('run_start',0,{}),('epoch',1,{'val_component_anchor_SROCC':.8,'train_processed_steps':2}),('final',1,{})]:
                w.writerow(dict(event=event,epoch=e,metrics_json=json.dumps(m)))
        output=self.root/'merged'
        calls=[]
        def train(model,loader,opt,scaler,device,cfg,epoch,optimizer_step_callback):
            for sample in loader:
                opt.zero_grad();model(sample).sum().backward();opt.step()
                optimizer_step_callback(True);calls.append('step')
            return {}
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv',['refit','--snapshot',str(snapshot),'--csv',str(source),'--output-dir',str(output)]))
            stack.enter_context(patch.object(runner,'build_dataloaders',return_value=self.loaders))
            stack.enter_context(patch.object(runner,'_build_single_dataset',side_effect=self.build))
            constructor=stack.enter_context(patch.object(runner,'CrossModalIQAJointPretrain',return_value=torch.nn.Linear(1,1)))
            stack.enter_context(patch.object(runner,'build_optimizer',side_effect=lambda c,m:torch.optim.SGD(m.parameters(),lr=.1)))
            stack.enter_context(patch.object(runner,'train_one_epoch',side_effect=train))
            stack.enter_context(patch.object(runner,'_extract_representation_state',side_effect=lambda state:state))
            stack.enter_context(patch('torch.cuda.is_available',return_value=False))
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner.main()
        self.assertEqual(calls,['step','step']);self.assertEqual(constructor.call_count,1)
        ckpt=torch.load(output/'final_merged_pretrain.pth',weights_only=False)
        self.assertEqual(ckpt['training_steps'],2)
        self.assertEqual(ckpt['protocol'],'pretrain_train_val_fixed_budget')
        with Path(str(output)+'.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        final=json.loads(rows[-1]['metrics_json'])
        self.assertEqual(final['validation_evaluations'],0);self.assertEqual(final['test_evaluations'],0)

if __name__=='__main__':unittest.main()
