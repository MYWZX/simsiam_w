"""Independent controls: no quality CE, MOS->PQ, shared PI/PT transfer."""
import copy
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import contextlib
import io
import unittest
from unittest.mock import patch
import torch
from models.two_stage_models import CrossModalIQAJointPretrain, CrossModalIQAFullModel, load_pretrained_backbone, simsiam_pair_loss
from train_cm_pretrain import _extract_representation_state, _quality_view_simsiam_loss, build_optimizer
from test_quality_route_ablations import config, Global, Local, Text
from utils.train_utils import load_yaml

ROOT=Path(__file__).resolve().parents[1]
def cfg(mode='base'):
    c=config();c['model']['quality_adapter']['enabled']=False
    c['model']['simsiam']={'proj_dim':8,'proj_hidden_dim':16,'pred_hidden_dim':4,'separate_quality_projection':True}
    c['model']['quality_auxiliary_heads']={'enabled':True,'components':['artifact','aesthetics'],'hidden_dim':16,'dropout':0.}
    c['train']['human_supervision_view']='three_view'
    if mode=='mos_pq':c['train']['quality_mos_via_pq']=True
    if mode=='shared_pi_pt':c['model']['shared_simsiam_projection']=True
    return c

def make(c,joint=True):
    with patch('models.two_stage_models.GlobalVisualEncoder',Global),patch('models.two_stage_models.ResNet50LocalEncoder',Local),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text):
        return (CrossModalIQAJointPretrain(c) if joint else CrossModalIQAFullModel(c)).eval()
def forward(m):
    return m._forward_three_views(*[torch.randn(3,3,16,16) for _ in range(3)],['cat']*3,['dog']*3)
def grad(loss,param):return torch.autograd.grad(loss,param,retain_graph=True,allow_unused=True)[0]

class ProjectionAblations(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_mos_heads_reach_pq_and_reuse_clean_projection(self):
        for mode in ['base','mos_pq']:
            m=make(cfg(mode));seen=[]
            h=m.quality_image_projector.register_forward_hook(lambda mod,ins,out:seen.append(out))
            out=forward(m);h.remove()
            self.assertEqual(len(seen),3) # Clean once + SafeAug twice
            for key in ['quality_pred_clean','quality_aux_artifact_pred','quality_aux_aesthetics_pred']:
                g=grad(out[key].sum(),m.quality_image_projector.net[0].weight)
                if mode=='mos_pq':self.assertIsNotNone(g);self.assertGreater(g.abs().sum().item(),0.)
                else:self.assertIsNone(g)
            self.assertIsNone(grad(out['align_pred_clean'].sum(),m.quality_image_projector.net[0].weight))
            torch.testing.assert_close(out['quality_clean_z'],torch.nn.functional.normalize(seen[0],dim=-1))

    def test_shared_pi_receives_image_relations_while_clean_quality_retains_pq(self):
        m=make(cfg('shared_pi_pt'));out=forward(m)
        pi=m.iqa.shared_image_projector.net[0].weight
        pq=m.quality_image_projector.net[0].weight
        pt=m.iqa.shared_text_projector.net[0].weight
        objectives=[_quality_view_simsiam_loss(out,torch.ones(3))]
        for name in ['prompt','answer']:
            objectives.append(simsiam_pair_loss(out['clean_p'],out[name+'_z'],out[name+'_p'],out['clean_z']))
        for loss in objectives:
            self.assertGreater(grad(loss,pi).abs().sum().item(),0.)
            self.assertIsNone(grad(loss,pq))
        self.assertGreater(grad(out['quality_clean_z'][:,0].sum(),pq).abs().sum().item(),0.)
        self.assertIsNone(grad(out['quality_clean_z'][:,0].sum(),pi))
        pp=simsiam_pair_loss(out['prompt_p'],out['answer_z'],out['answer_p'],out['prompt_z'])
        self.assertGreater(grad(pp,pt).abs().sum().item(),0.)
        self.assertIsNone(grad(pp,pi)) # Prompt-Answer is text-only
        region=m.project_patch_regions(out['align_patch_features_clean'],torch.ones(3,4))
        self.assertGreater(grad(region['z_proj'][:,0].sum(),pi).abs().sum().item(),0.)
        self.assertFalse(hasattr(m,'image_projector'));self.assertFalse(hasattr(m,'text_projector'))
        self.assertEqual(sum(p is pi for p in m.parameters()),1)

    def test_shared_downstream_orders_and_complete_strict_transfer(self):
        c=cfg('shared_pi_pt');source=make(c)
        dc=copy.deepcopy(c);dc['model']['quality_adapter']['enabled']=True
        dc['model'].update(quality_global_feature='g',align_global_feature='g')
        target=make(dc,joint=False)
        state=_extract_representation_state(source.state_dict())
        def load(s):
            with patch('torch.load',return_value={'model':s}):
                load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        load(state)
        for prefix in ['shared_image_projector','shared_text_projector']:
            for k,v in getattr(source.iqa,prefix).state_dict().items():
                torch.testing.assert_close(v,getattr(target,prefix).state_dict()[k])
        torch.testing.assert_close(source.quality_image_projector.net[0].weight,target.quality_adapter.net[0].weight)
        self.assertNotEqual(target.quality_adapter.net[0].weight.data_ptr(),target.shared_image_projector.net[0].weight.data_ptr())
        seen={}
        hooks=[getattr(target,n).register_forward_pre_hook(lambda mod,ins,n=n:seen.update({n:ins[0]})) for n in ['align_global_adapter','align_patch_adapter','align_text_adapter']]
        image=torch.randn(3,3,16,16);out=target.forward_single(image,['cat']*3)
        for h in hooks:h.remove()
        g=target.backbone.fusion.semantic_proj(out['f_sem'])
        torch.testing.assert_close(seen['align_global_adapter'],target.project_shared_image(g))
        tokens=target._make_patch_tokens(out['f_loc'])
        torch.testing.assert_close(seen['align_patch_adapter'],target.project_shared_image(tokens))
        torch.testing.assert_close(seen['align_text_adapter'],torch.nn.functional.normalize(target.shared_text_projector(out['text_fused']),dim=-1))
        self.assertGreater(grad(out['s_align'].sum(),target.shared_image_projector.net[0].weight).abs().sum().item(),0.)
        self.assertIsNone(grad(out['s_quality'].sum(),target.shared_image_projector.net[0].weight))
        for key in ['iqa.shared_image_projector.net.0.weight','iqa.shared_text_projector.net.0.weight']:
            bad=dict(state);bad.pop(key)
            with self.assertRaisesRegex(RuntimeError,'Incomplete shared'):load(bad)
        old={k:v for k,v in state.items() if 'shared_' not in k}
        with self.assertRaisesRegex(RuntimeError,'architecture mismatch'):load(old)
        legacy=make(cfg(),joint=False)
        with patch('torch.load',return_value={'model':state}):
            with self.assertRaisesRegex(RuntimeError,'architecture mismatch'):load_pretrained_backbone(legacy,'mock.pth',load_scope='representation')
        opt=build_optimizer(c,source)
        ids=[id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertIn(id(source.iqa.shared_image_projector.net[0].weight),ids)

    def test_real_three_view_training_step_and_evaluation_with_pcgrad(self):
        from train_cm_pretrain import train_one_epoch, evaluate
        for mode in ['base','mos_pq','shared_pi_pt']:
            c=cfg(mode)
            c['train'].update(use_joint_iqa_pretraining=True,amp=False,pcgrad={'enabled':True},
                lambda_quality_reg=.2,lambda_align_pretrain_reg=.2,
                lambda_img_siam=.15,lambda_img_prompt=.25,lambda_img_answer=.2,lambda_prompt_answer=.05,
                quality_component_anchor_ordinal={'enabled':True,'num_levels':5},
                lambda_artifact_anchor=.01,lambda_artifact_ordinal=.015,
                lambda_aesthetics_anchor=.01,lambda_aesthetics_ordinal=.015)
            m=make(c);opt=build_optimizer(c,m)
            before=m.quality_image_projector.net[0].weight.detach().clone()
            pi_before=m.iqa.shared_image_projector.net[0].weight.detach().clone() if mode=='shared_pi_pt' else None
            batch={k:torch.randn(3,3,16,16) for k in ['image1','image2','image_clean']}
            batch.update(prompt=['cat']*3,answer=['dog']*3,
                mos=torch.tensor([.2,.5,.8]),quality_score=torch.tensor([.2,.5,.8]),
                align_score=torch.tensor([.3,.6,.7]),artifact_score=torch.tensor([.2,.4,.8]),aesthetics_score=torch.tensor([.3,.6,.9]))
            with contextlib.redirect_stdout(io.StringIO()):
                result=train_one_epoch(m,[batch],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,1)
                metrics=evaluate(m,[batch],torch.device('cpu'),c)
            self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
            self.assertTrue(torch.isfinite(torch.tensor(metrics['loss'])))
            self.assertFalse(torch.equal(before,m.quality_image_projector.net[0].weight))
            if pi_before is not None:self.assertFalse(torch.equal(pi_before,m.iqa.shared_image_projector.net[0].weight))

    def test_noce_removes_bank_and_zeroes_actual_losses_but_preserves_pq(self):
        from train_cm_iqa import compute_losses
        from train_cm_pretrain import _compute_quality_text_loss
        c=cfg();c['model']['quality_text_bank']={'enabled':False};c['train']['lambda_quality_text_ce']=0.
        source=make(c);out=forward(source)
        self.assertIsNone(source.iqa.quality_text_bank)
        self.assertNotIn('quality_text_logits',out)
        zero=_compute_quality_text_loss(out,torch.tensor([.2,.5,.8]),torch.ones(3),None)
        self.assertEqual(zero['quality_text_ce'].item(),0.)
        self.assertGreater(grad(out['quality_clean_z'][:,0].sum(),source.quality_image_projector.net[0].weight).abs().sum().item(),0.)
        dc=copy.deepcopy(c);dc['model']['quality_adapter']['enabled']=True
        model=make(dc,joint=False)
        batch={'image':torch.randn(3,3,16,16),'prompt':['cat']*3,'score':torch.tensor([.2,.5,.8]),'quality_score':torch.tensor([.2,.5,.8]),'align_score':torch.tensor([.3,.6,.7])}
        result,losses=compute_losses(model,batch,'cpu',dc)
        self.assertNotIn('quality_text_logits',result)
        self.assertEqual(losses['quality_text_ce'].item(),0.)
        self.assertEqual(losses['objective_quality_text'].item(),0.)
        losses['loss'].backward()

    def test_twelve_effective_main_configs_do_not_reenable_ce_or_change_baseline(self):
        import train_cm_pretrain as pre
        import train_cm_iqa as down
        class Captured(Exception):pass
        for mode in ['noce','mos_pq','shared_pi_pt']:
            for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
                raw=subprocess.check_output(['bash',str(ROOT/'scripts/run_projection_ablation.sh'),mode,stage,'--dry-run'],text=True)
                parts=shlex.split(raw);idx=next(i for i,v in enumerate(parts) if v in ['train_cm_pretrain.py','train_cm_iqa.py'])
                argv=parts[idx:];module=pre if stage=='pretrain' else down
                seen=[]
                def snapshot(config,*a,**kw):seen.append(copy.deepcopy(config));raise Captured()
                with tempfile.TemporaryDirectory() as d:
                    argv[argv.index('--output-dir')+1]=d
                    argv[argv.index('--results-csv')+1]=d+'/results.csv'
                    with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(Captured):module.main()
                c=seen[0]
                self.assertEqual(c['seed'],42);self.assertEqual(c['model']['global_encoder']['output_dim'],768)
                self.assertEqual(c['model']['text_encoder']['learnable_context_tokens'],4)
                self.assertEqual(c['model']['quality_text_bank']['enabled'],mode!='noce')
                self.assertEqual(c['train']['lambda_quality_text_ce'],0. if mode=='noce' else .005)
                self.assertEqual(c['model'].get('shared_simsiam_projection',False),mode=='shared_pi_pt')
                if stage=='pretrain':
                    self.assertTrue(c['train']['pcgrad']['enabled'])
                    self.assertEqual(c['model']['local_encoder'].get('feature_stage','layer4'),'layer4')
                    self.assertEqual(c['train'].get('quality_mos_via_pq',False),mode=='mos_pq')
                else:
                    self.assertEqual(c['model']['quality_global_feature'],'g');self.assertEqual(c['model']['align_global_feature'],'g')
                    self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3')
                    self.assertTrue(c['train']['reinit_local_projection'])
                    self.assertEqual(c['train']['layerwise_lr']['global_encoder_mult'],.02)
                    self.assertEqual(c['model']['quality_adapter']['mode'],'residual_feature')

    def test_presets_and_launcher_keep_controls_independent(self):
        for mode in ['noce','mos_pq','shared_pi_pt']:
            p=load_yaml(str(ROOT/f'configs/projection_ablations/pretrain_{mode}.yaml'))
            self.assertEqual(p['train']['pcgrad']['enabled'],True)
            self.assertEqual(p['train'].get('quality_mos_via_pq',False),mode=='mos_pq')
            self.assertEqual(p['model'].get('shared_simsiam_projection',False),mode=='shared_pi_pt')
            self.assertEqual(p['model']['quality_text_bank']['enabled'],mode!='noce')
            self.assertEqual(p['train']['lambda_quality_text_ce'],0. if mode=='noce' else .005)
            for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
                command=subprocess.check_output(['bash',str(ROOT/'scripts/run_projection_ablation.sh'),mode,stage,'--dry-run'],text=True)
                self.assertNotIn('--quality-text-levels',command)
                if mode=='noce':self.assertNotIn('--quality-text-blend',command)
                if stage!='pretrain':
                    d=load_yaml(str(ROOT/f'configs/projection_ablations/{stage}_{mode}.yaml'))
                    self.assertEqual(d['model']['quality_global_feature'],'g');self.assertEqual(d['model']['align_global_feature'],'g')
                    self.assertNotIn('skip_pretrained_quality_adapter',d['train'])
                    self.assertIn('--reinit-local-projection',command)

if __name__=='__main__':unittest.main()
