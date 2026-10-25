"""Global/local quality supervision, matched transfer and real train integration."""
import contextlib
import copy
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from models.two_stage_models import DUAL_QUALITY_TRANSFER_PREFIXES, load_pretrained_backbone
from train_cm_pretrain import (_extract_representation_state, _compute_quality_component_anchor_ordinal_losses,
    _compute_quality_text_loss, train_one_epoch, evaluate, build_optimizer)
from train_cm_iqa import compute_losses, evaluate as downstream_evaluate
from test_projection_ablations import cfg as original_cfg, make
from test_quality_route_ablations import Bank, grads
from utils.quality_representation import dual_quality_diagnostics
from utils.train_utils import load_yaml

ROOT=Path(__file__).resolve().parents[1]

def config():
    c=original_cfg();m=c['model']
    m['dual_quality_representation']={'enabled':True,'hidden_dim':16,'dropout':0.,'feature_local_init':.5}
    m['quality_global_feature']='g';m['align_global_feature']='g'
    m['quality_adapter'].update(enabled=True,mode='replace',normalize=True)
    c['train'].update(use_joint_iqa_pretraining=True,amp=False,pcgrad={'enabled':True},
        lambda_quality_reg=.2,lambda_align_pretrain_reg=.2,
        lambda_img_siam=.15,lambda_img_prompt=.25,lambda_img_answer=.2,lambda_prompt_answer=.05,
        quality_component_anchor_ordinal={'enabled':True,'num_levels':5},
        lambda_artifact_anchor=.01,lambda_artifact_ordinal=.015,
        lambda_aesthetics_anchor=.01,lambda_aesthetics_ordinal=.015)
    return c

def batch():
    b={k:torch.randn(3,3,16,16) for k in ['image1','image2','image_clean']}
    b.update(prompt=['cat']*3,answer=['dog']*3,mos=torch.tensor([.2,.5,.8]),
        quality_score=torch.tensor([.2,.5,.8]),align_score=torch.tensor([.3,.6,.7]),
        artifact_score=torch.tensor([.2,.4,.8]),aesthetics_score=torch.tensor([.3,.6,.9]))
    return b

def views(m,b):return m.forward_views(b['image1'],b['image2'],b['prompt'],b['answer'],image_clean=b['image_clean'])

def gradient(loss,p):return torch.autograd.grad(loss,p,retain_graph=True,allow_unused=True)[0]

class DualQualityTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_forward_formula_and_anchors_ordinal_ce_reach_both_visual_encoders(self):
        c=config();m=make(c);m.iqa.quality_text_bank=Bank();b=batch()
        captured=[];h=m.iqa.quality_adapter.register_forward_pre_hook(lambda mod,args:captured.append(args[0]))
        out=views(m,b);h.remove()
        for feature,image in zip(captured,[b['image_clean'],b['image1'],b['image2']]):
            expected=m.iqa.backbone.fusion.semantic_proj(m.iqa.backbone.global_encoder(image))
            torch.testing.assert_close(feature,expected)
        self.assertEqual(len(captured),3) # no extra independent PQ or Clean BN pass
        self.assertIsNone(m.quality_image_projector)
        qg=out['quality_global_clean_z'];ql=out['quality_local_clean_z']
        torch.testing.assert_close(out['quality_clean_z'],torch.nn.functional.normalize(.5*qg+.5*ql,dim=-1))
        torch.testing.assert_close(m.iqa.quality_text_bank.seen,out['quality_clean_z'])
        anchor,_=_compute_quality_component_anchor_ordinal_losses(out,b,torch.ones(3),c['train'])
        ce=_compute_quality_text_loss(out,b['quality_score'],torch.ones(3),m.iqa.quality_text_bank.centers)
        objectives={k:v for k,v in anchor.items() if k.endswith(('anchor_ce','ordinal_bce'))}
        self.assertEqual(len(objectives),4)
        objectives['quality_ce']=ce['quality_text_ce']
        parameters=[m.iqa.backbone.global_encoder.proj.weight,m.iqa.backbone.local_encoder.proj.weight,
            m.iqa.quality_adapter.net[0].weight,m.iqa.quality_local_encoder.net[1].weight,
            m.iqa.nss_gate[0].weight,m.iqa.quality_feature_mix_logit]
        for name,loss in objectives.items():
            for p in parameters:
                with self.subTest(loss=name,shape=p.shape):
                    g=gradient(loss,p);self.assertIsNotNone(g);self.assertGreater(g.abs().sum().item(),0.)
        # The auxiliary objectives reach ResNet through the quality local path,
        # without accidentally training the discarded Fusion attention path.
        self.assertIsNone(gradient(ce['quality_text_ce'],m.iqa.backbone.fusion.attn.in_proj_weight))
        self.assertEqual(dual_quality_diagnostics(m)['quality_feature_local_weight'],.5)

    def test_global_and_local_scores_keep_separate_inputs_and_align_is_unchanged(self):
        c=config();m=make(c,joint=False)
        legacy=copy.deepcopy(c);legacy['model'].pop('dual_quality_representation')
        torch.manual_seed(42);base=make(legacy,joint=False)
        image=torch.randn(3,3,16,16);out=m.forward_single(image,['cat']*3);old=base.forward_single(image,['cat']*3)
        torch.testing.assert_close(out['s_align'],old['s_align'])
        torch.testing.assert_close(out['s_quality_global'],torch.sigmoid(m.global_quality_head(out['z_quality_global']).squeeze(-1)))
        torch.testing.assert_close(out['s_quality_patch'],torch.sigmoid(m.patch_quality_head(out['quality_local_feature']).squeeze(-1)))
        torch.testing.assert_close(out['quality_local_feature'],torch.nn.functional.normalize((out['quality_patch_weights'].unsqueeze(-1)*out['patch_tokens']).sum(1),dim=-1))
        beta=out['quality_mix'];torch.testing.assert_close(out['s_quality'],(1-beta)*out['s_quality_global']+beta*out['s_quality_patch'])
        self.assertIsNone(gradient(out['s_quality_global'].sum(),m.backbone.local_encoder.proj.weight))
        self.assertIsNone(gradient(out['s_quality_patch'].sum(),m.backbone.global_encoder.proj.weight))
        self.assertIsNone(gradient(out['s_quality'].sum(),m.quality_feature_mix_logit))
        self.assertNotEqual(m.quality_feature_mix_logit.data_ptr(),m.quality_mix_logit.data_ptr())

    def test_complete_local_representation_transfer_and_corruption_failures(self):
        c=config();source=make(c);target=make(c,joint=False)
        # Make transfer observable independently of seeded initialization.
        with torch.no_grad():
            source.iqa.nss_gate[0].weight.add_(.2)
            source.iqa.quality_feature_mix_logit.fill_(.7)
        state=_extract_representation_state(source.state_dict())
        head_before=target.global_quality_head[0].weight.detach().clone()
        def load(s,**kwargs):
            with patch('torch.load',return_value={'model':s}):load_pretrained_backbone(target,'mock.pth',load_scope='representation',**kwargs)
        load(state)
        for key,value in source.iqa.state_dict().items():
            if key.startswith(DUAL_QUALITY_TRANSFER_PREFIXES):torch.testing.assert_close(value,target.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(head_before,target.global_quality_head[0].weight)
        for key in ['iqa.nss_gate.0.weight','iqa.local_patch_proj.weight','iqa.quality_local_encoder.net.1.weight',
                    'iqa.nss_proj.0.bias','iqa.backbone.local_encoder.proj.bias','iqa.quality_adapter.net.1.running_var','iqa.quality_feature_mix_logit','iqa.dual_quality_signature']:
            bad=dict(state);bad.pop(key)
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'Incomplete dual-quality|lacks local projection'):load(bad)
        bad=dict(state);bad['iqa.quality_local_encoder.net.1.weight']=torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError,'Incomplete dual-quality|lacks local projection'):load(bad)
        for kwargs in [{'reinit_local_projection':True},{'skip_quality_adapter':True}]:
            with self.assertRaisesRegex(ValueError,'complete matched-stage'):load(state,**kwargs)
        legacy=copy.deepcopy(c);legacy['model'].pop('dual_quality_representation');old=make(legacy)
        with self.assertRaisesRegex(RuntimeError,'architecture mismatch'):load(_extract_representation_state(old.state_dict()))
        old_target=make(legacy,joint=False)
        with patch('torch.load',return_value={'model':state}),self.assertRaisesRegex(RuntimeError,'architecture mismatch'):
            load_pretrained_backbone(old_target,'mock.pth',load_scope='representation')

    def test_real_pcgrad_training_downstream_ce_and_full_state_roundtrip(self):
        c=config();m=make(c);m.iqa.quality_text_bank=Bank();b=batch();opt=build_optimizer(c,m)
        before=m.iqa.quality_local_encoder.net[1].weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            result=train_one_epoch(m,[b],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,1)
            metrics=evaluate(m,[b],torch.device('cpu'),c)
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
        self.assertFalse(torch.equal(before,m.iqa.quality_local_encoder.net[1].weight))
        self.assertIn('quality_feature_local_weight',metrics)
        clone=make(c);clone.iqa.quality_text_bank=Bank();clone.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(views(m,b)['quality_clean_z'],views(clone,b)['quality_clean_z'])
        target=make(c,joint=False);target.quality_text_bank=Bank()
        down={'image':b['image_clean'],'prompt':b['prompt'],'score':b['quality_score'],
              'quality_score':b['quality_score'],'align_score':b['align_score']}
        out,losses=compute_losses(target,down,'cpu',c)
        torch.testing.assert_close(target.quality_text_bank.seen,out['quality_representation'])
        self.assertGreater(gradient(losses['objective_quality_text'],target.backbone.local_encoder.proj.weight).abs().sum().item(),0.)
        self.assertGreater(gradient(losses['objective_quality_text'],target.quality_adapter.net[0].weight).abs().sum().item(),0.)
        diag=downstream_evaluate(target,[down],'cpu',c)
        self.assertIn('val_quality_feature_local_weight',diag)
        ids=[id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertIn(id(m.iqa.quality_feature_mix_logit),ids)

    def test_all_effective_main_configs_use_matched_layer3_and_no_residual(self):
        import train_cm_pretrain as pre
        import train_cm_iqa as down
        class Captured(Exception):pass
        for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
            command=subprocess.check_output(['bash',str(ROOT/'scripts/run_dual_quality.sh'),stage,'--dry-run'],text=True)
            self.assertNotIn('--reinit-local-projection',command)
            self.assertNotIn('residual_feature',command)
            parts=shlex.split(command);idx=next(i for i,x in enumerate(parts) if x in ['train_cm_pretrain.py','train_cm_iqa.py'])
            argv=parts[idx:];module=pre if stage=='pretrain' else down;seen=[]
            def snapshot(c,*args,**kwargs):seen.append(copy.deepcopy(c));raise Captured()
            with tempfile.TemporaryDirectory() as d:
                argv[argv.index('--output-dir')+1]=d;argv[argv.index('--results-csv')+1]=d+'/results.csv'
                with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Captured):module.main()
            c=seen[0]
            self.assertTrue(c['model']['dual_quality_representation']['enabled'])
            self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3')
            self.assertEqual(c['model']['quality_global_feature'],'g')
            self.assertEqual(c['model']['quality_adapter']['mode'],'replace')
            self.assertEqual(c['model']['global_encoder']['output_dim'],768)
            self.assertEqual(c['train']['lambda_quality_text_ce'],.005)
            self.assertFalse(c['train'].get('reinit_local_projection',False))

    def test_invalid_global_only_or_parallel_modes_are_rejected(self):
        cases=[('quality_global_feature','z'),('quality_no_local_branch',True),('quality_branch_use_text_gate',True),('shared_simsiam_projection',True)]
        for key,value in cases:
            c=config();c['model'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):make(c)
        c=config();c['model']['quality_adapter']['mode']='parallel_score'
        with self.assertRaisesRegex(ValueError,'replace or residual_feature mode'):make(c)

if __name__=='__main__':unittest.main()
