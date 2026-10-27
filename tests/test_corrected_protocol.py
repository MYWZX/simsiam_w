import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from utils.data_protocols import fit_1k_train_scaler, loader_rng_kwargs
from utils.refit_protocol import (make_plan, load_plan, merge_train_validation,
    run_fixed_budget_refit, step_lrs)
from train_cm_iqa import get_selection_value

class Samples(Dataset):
    def __init__(self):self.df=pd.DataFrame({'name':[f'{i}.png' for i in range(10)],'MOS':list(range(10))})
    def __len__(self):return 10
    def __getitem__(self,i):return torch.tensor([float(i)])

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.ckpt=self.root/'source.pth';self.ckpt.write_bytes(b'weights')
        self.cfg={'seed':42,'model':{'clip_quick_gelu':True},'dataset':{'type':'agiqa1k_iqa','mos_normalization':'train_minmax'},'loader':{'batch_size':2,'num_workers':0,'isolate_rng':True},'train':{'protocol_version':'corrected_v2','epochs':3,'pretrained_checkpoint':str(self.ckpt),'selection_metric':'val_quality_SROCC','selection_mode':'max'}}
        self.base=Samples();self.eval=Samples()
        self.loaders=tuple(DataLoader(Subset(base,idx),batch_size=2,drop_last=name=='train') for name,base,idx in [('train',self.base,list(range(6))),('val',self.eval,[6,7]),('test',self.eval,[8,9])])
        self.csv=self.root/'dev.csv'
        with self.csv.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            for event,e,m in [('run_start',0,{}),('epoch',1,{'val_val_quality_SROCC':.5,'train_processed_steps':3,'test_SROCC':1.}),('epoch',2,{'val_val_quality_SROCC':.8,'train_processed_steps':3,'test_SROCC':0.}),('epoch',3,{'val_val_quality_SROCC':.6,'train_processed_steps':3}),('final',3,{'best_test':{'quality_SROCC':1.}})]:w.writerow(dict(event=event,epoch=e,metrics_json=json.dumps(m)))
    def plan(self):return make_plan(self.cfg,self.csv,self.loaders,get_selection_value)
    def test_scaler_ignores_heldout_labels(self):
        fit_1k_train_scaler(self.base,self.eval,range(6),self.cfg)
        self.assertEqual((self.base.min_mos,self.eval.max_mos),(0.,5.))
        self.base.df.loc[9,'MOS']=99999
        fit_1k_train_scaler(self.base,self.eval,range(6),self.cfg)
        self.assertEqual(self.base.max_mos,5.)
    def test_plan_selection_ignores_test(self):
        plan=self.plan();self.assertEqual(plan['total_steps'],6);self.assertEqual(plan['validation_selection']['epoch'],2)
    def test_plan_integrity_and_config(self):
        p=self.root/'plan.json';p.write_text(json.dumps(self.plan()))
        load_plan(p,self.cfg)
        c=copy.deepcopy(self.cfg);c['train']['epochs']=4
        with self.assertRaisesRegex(ValueError,'config'):load_plan(p,c)
        self.ckpt.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source changed'):load_plan(p,self.cfg)
    def test_merge_preserves_test_and_uses_train_transforms(self):
        plan=self.plan();test_before=list(self.loaders[2].dataset.indices)
        merged,val=merge_train_validation(*self.loaders,self.cfg,plan)
        self.assertIsNone(val);self.assertIs(merged.dataset.dataset,self.base)
        self.assertEqual(merged.dataset.indices,list(range(8)))
        self.assertEqual(self.loaders[2].dataset.indices,test_before)
        self.assertEqual(self.base.max_mos,7.);self.assertEqual(self.eval.max_mos,7.)
        self.assertEqual(self.cfg['dataset']['refit_split_manifest']['test_count'],2)
    def test_changed_split_rejected(self):
        plan=self.plan();self.eval.df.loc[9,'name']='new.png'
        with self.assertRaises(ValueError):merge_train_validation(*self.loaders,self.cfg,plan)
    def test_rng_isolation(self):
        def order(consume):
            loader=DataLoader(range(20),batch_size=4,shuffle=True,**loader_rng_kwargs(self.cfg,'train'))
            torch.randn(consume)
            list(DataLoader(range(3),**loader_rng_kwargs(self.cfg,'test')))
            return torch.cat(list(loader)).tolist()
        self.assertEqual(order(1),order(100))
    def test_exact_budget_final_test_once(self):
        plan=self.plan();merged,_=merge_train_validation(*self.loaders,self.cfg,plan)
        model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.1)
        calls=[];logs=[]
        def train(model,loader,opt,scaler,device,cfg,epoch,optimizer_step_callback):
            for batch in loader:
                opt.zero_grad();model(batch).sum().backward();opt.step();optimizer_step_callback(True);calls.append('step')
            return {}
        def evaluate(*args,**kwargs):calls.append(kwargs['split']);return {'test_quality_SROCC':.5}
        run_fixed_budget_refit(model=model,train_loader=merged,test_loader=self.loaders[2],optimizer=opt,device=torch.device('cpu'),cfg=self.cfg,plan=plan,logger=SimpleNamespace(log=lambda **kw:logs.append(kw)),train_epoch=train,evaluate_fn=evaluate,metadata_path='meta.json')
        self.assertEqual(calls,['step']*6+['test']);self.assertEqual(logs[-1]['metrics']['training_steps'],6)
        self.assertEqual([x['epoch'] for x in logs if x['event']=='refit_epoch'],[1,2])
        self.assertEqual(logs[-1]['metrics']['optimizer_updates'],6)
    def test_schedule_preserves_source_epoch_units(self):
        self.assertEqual(step_lrs([.1],.001,2,3,60),[.1])
        self.assertGreater(step_lrs([.1],.001,6,3,60)[0],.09)


class ModelProtocolTests(unittest.TestCase):
    def test_activation_marker_roundtrip_and_mix_rejected(self):
        from unittest.mock import patch
        from test_dual_quality_residual import residual_config
        from test_projection_ablations import make
        from train_cm_pretrain import _extract_representation_state
        from models.two_stage_models import load_pretrained_backbone
        c=residual_config();c['model']['clip_quick_gelu']=True
        src=make(c);dst=make(c,joint=False)
        state=_extract_representation_state(src.state_dict())
        self.assertIn('iqa.backbone.clip_activation_signature',state)
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(dst,'mock',load_scope='representation')
        legacy=make(residual_config(),joint=False)
        with patch('torch.load',return_value={'model':state}),self.assertRaisesRegex(RuntimeError,'activation mismatch'):
            load_pretrained_backbone(legacy,'mock',load_scope='representation')
        state.pop('iqa.backbone.clip_activation_signature')
        with patch('torch.load',return_value={'model':state}),self.assertRaisesRegex(RuntimeError,'activation mismatch'):
            load_pretrained_backbone(dst,'mock',load_scope='representation')
    def test_projection_lr_group_only(self):
        from test_dual_quality_residual import residual_config
        from test_projection_ablations import make
        from train_cm_iqa import build_optimizer
        c=residual_config();c['model']['shared_simsiam_projection']=True
        c['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'shared_projection_mult':.1})
        m=make(c,joint=False);opt=build_optimizer(c,m)
        groups={g['name']:g for g in opt.param_groups}
        self.assertAlmostEqual(groups['shared_projection']['lr'],1e-5)
        expected={id(p) for n,p in m.named_parameters() if n.startswith(('shared_image_projector.','shared_text_projector.')) and p.requires_grad}
        self.assertEqual(expected,{id(p) for p in groups['shared_projection']['params']})
        self.assertAlmostEqual(groups['quality_adapter']['lr'],1e-4)
    def test_clip_activation_forwarded_to_both_factories(self):
        from unittest.mock import patch
        from test_quality_route_ablations import Global,Local,Text,config
        from models.two_stage_models import ImageTextBackbone
        c=config();c['model']['clip_quick_gelu']=True
        with patch('models.two_stage_models.GlobalVisualEncoder',side_effect=Global) as visual,patch('models.two_stage_models.ResNet50LocalEncoder',Local),patch('models.two_stage_models.FrozenCLIPTextEncoder',side_effect=Text) as text:
            ImageTextBackbone(c)
        self.assertTrue(visual.call_args.kwargs['force_quick_gelu'])
        self.assertTrue(text.call_args.kwargs['force_quick_gelu'])
    def test_bn_policy_keeps_resnet_affine_trainable(self):
        from unittest.mock import patch
        from test_quality_route_ablations import Global,Local,Text,config
        from models.two_stage_models import ImageTextBackbone
        class WithBN(Local):
            def __init__(self,**kwargs):super().__init__(**kwargs);self.bn=torch.nn.BatchNorm2d(4)
        c=config();c['model']['local_encoder']['freeze_bn_stats']=True
        with patch('models.two_stage_models.GlobalVisualEncoder',Global),patch('models.two_stage_models.ResNet50LocalEncoder',WithBN),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text):m=ImageTextBackbone(c)
        m.train();self.assertFalse(m.local_encoder.bn.training);self.assertTrue(m.local_encoder.bn.weight.requires_grad)

class ConfigIsolationTests(unittest.TestCase):
    def test_each_control_changes_only_intended_field(self):
        import yaml
        root=Path(__file__).resolve().parents[1]/'configs/protocol_v2'
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            base=yaml.safe_load((root/f'{ds}_baseline.yaml').read_text())
            for mode in ['align_aux','proj_lr','qa_bn','resnet_bn']:
                variant=yaml.safe_load((root/f'{ds}_{mode}.yaml').read_text())
                for k in ['run_id','output_dir','results_csv']:variant[k]=base[k]
                expected=copy.deepcopy(base)
                if mode=='align_aux':expected['train']['lambda_align_aux']=0. if ds=='agiqa1k' else .1
                if mode=='proj_lr':expected['train']['layerwise_lr']['shared_projection_mult']=.1
                if mode=='qa_bn':expected['model']['quality_adapter']['freeze_bn_stats']=True
                if mode=='resnet_bn':expected['model']['local_encoder']['freeze_bn_stats']=True
                self.assertEqual(variant,expected)
            self.assertTrue(base['model']['clip_quick_gelu'])
            self.assertEqual(base['model']['global_encoder']['model_name'],'ViT-L-14')
            self.assertEqual(base['model']['local_encoder']['feature_stage'],'layer3')

class MainRefitTests(unittest.TestCase):
    def test_cli_refit_bypasses_selection_and_evaluates_test_once(self):
        from contextlib import ExitStack,redirect_stdout
        from unittest.mock import patch
        import io
        import train_cm_iqa as training
        fixture=ProtocolTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        cfg=fixture.cfg;cfg.update(output_dir=str(fixture.root/'refit'),results_csv=str(fixture.root/'refit.csv'),run_id='refit')
        plan=fixture.plan();path=fixture.root/'plan.json';path.write_text(json.dumps(plan))
        calls=[]
        def train(model,loader,optimizer,scaler,device,cfg,epoch,optimizer_step_callback):
            for _ in loader:optimizer_step_callback(True);calls.append('step')
            return {}
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv',['train_cm_iqa.py','--config','unused','--refit-plan',str(path)]))
            stack.enter_context(patch.object(training,'load_yaml',return_value=cfg))
            stack.enter_context(patch.object(training,'build_dataloaders',return_value=fixture.loaders))
            stack.enter_context(patch.object(training,'build_iqa_model',return_value=torch.nn.Linear(1,1)))
            stack.enter_context(patch.object(training,'load_pretrained_backbone'))
            stack.enter_context(patch.object(training,'build_optimizer',side_effect=lambda cfg,m:torch.optim.SGD(m.parameters(),lr=.1)))
            stack.enter_context(patch.object(training,'train_one_epoch',side_effect=train))
            evaluate=stack.enter_context(patch.object(training,'evaluate',return_value={'test_quality_SROCC':.5}))
            stack.enter_context(patch('torch.cuda.is_available',return_value=False))
            stack.enter_context(patch('torch.save',side_effect=AssertionError('No downstream disk weights')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            training.main()
        self.assertEqual(calls,['step']*6)
        self.assertEqual(evaluate.call_count,1);self.assertEqual(evaluate.call_args.kwargs['split'],'test')
        with open(cfg['results_csv']) as f:rows=list(csv.DictReader(f))
        final=json.loads(rows[-1]['metrics_json'])
        self.assertEqual(final['protocol'],'train_val_fixed_budget');self.assertEqual(final['training_steps'],6)

if __name__=='__main__':unittest.main()
