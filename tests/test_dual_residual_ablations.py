"""Projection ablations on the current matched layer3 dual residual baseline."""
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
from models.two_stage_models import load_pretrained_backbone, simsiam_pair_loss
from train_cm_pretrain import _extract_representation_state, _quality_view_simsiam_loss
from train_cm_iqa import compute_losses
from test_dual_quality_residual import residual_config
from test_dual_quality_representation import batch, views, gradient
from test_projection_ablations import make
from utils.train_utils import load_yaml

ROOT=Path(__file__).resolve().parents[1]
class DualResidualAblations(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_shared_pi_moves_ii_away_from_independent_quality_encoder(self):
        c=residual_config();c['model']['shared_simsiam_projection']=True
        source=make(c);b=batch();captured=[]
        h=source.iqa.shared_image_projector.register_forward_pre_hook(lambda m,args:captured.append(args[0]))
        out=views(source,b);h.remove()
        ii=_quality_view_simsiam_loss(out,torch.ones(3))
        pi=source.iqa.shared_image_projector.net[0].weight;pq=source.iqa.quality_adapter.net[0].weight
        self.assertGreater(gradient(ii,pi).abs().sum().item(),0.)
        self.assertIsNone(gradient(ii,pq))
        for name in ['prompt','answer']:
            loss=simsiam_pair_loss(out['clean_p'],out[name+'_z'],out[name+'_p'],out['clean_z'])
            self.assertGreater(gradient(loss,pi).abs().sum().item(),0.)
            self.assertIsNone(gradient(loss,pq))
        # I-I still receives g, matching the baseline operand; only projector changes.
        for seen,image in zip(captured[-2:],[b['image1'],b['image2']]):
            expected=source.iqa.backbone.fusion.semantic_proj(source.iqa.backbone.global_encoder(image))
            torch.testing.assert_close(seen,expected)
        self.assertGreater(gradient(out['quality_clean_z'][:,0].sum(),pq).abs().sum().item(),0.)
        self.assertIsNone(gradient(out['quality_clean_z'][:,0].sum(),pi))
        self.assertIsNone(source.quality_image_projector) # PQ is persistent iqa.quality_adapter
        target=make(c,joint=False);state=_extract_representation_state(source.state_dict())
        with patch('torch.load',return_value={'model':state}):load_pretrained_backbone(target,'mock.pth',load_scope='representation')
        for name in ['shared_image_projector','shared_text_projector','quality_adapter','quality_local_encoder']:
            for key,value in getattr(source.iqa,name).state_dict().items():torch.testing.assert_close(value,getattr(target,name).state_dict()[key])
        self.assertNotEqual(target.shared_image_projector.net[0].weight.data_ptr(),target.quality_adapter.net[0].weight.data_ptr())
        pred=target.forward_single(b['image_clean'],b['prompt'])
        self.assertGreater(gradient(pred['s_align'].sum(),target.shared_image_projector.net[0].weight).abs().sum().item(),0.)
        self.assertIsNone(gradient(pred['s_quality'].sum(),target.shared_image_projector.net[0].weight))

    def test_baseline_quality_mos_and_component_heads_already_update_pq(self):
        source=make(residual_config());out=views(source,batch())
        pq=source.iqa.quality_adapter.net[0].weight
        for name in ['quality_pred_clean','quality_aux_artifact_pred','quality_aux_aesthetics_pred']:
            g=gradient(out[name].sum(),pq)
            self.assertIsNotNone(g);self.assertGreater(g.abs().sum().item(),0.)

    def test_noce_actual_loss_is_zero_and_both_quality_paths_remain(self):
        c=residual_config();c['model']['quality_text_bank']={'enabled':False};c['train']['lambda_quality_text_ce']=0.
        source=make(c);b=batch();out=views(source,b)
        self.assertNotIn('quality_text_logits',out)
        for p in [source.iqa.quality_adapter.net[0].weight,source.iqa.backbone.local_encoder.proj.weight]:
            self.assertGreater(gradient(out['quality_clean_z'][:,0].sum(),p).abs().sum().item(),0.)
        target=make(c,joint=False)
        down={'image':b['image_clean'],'prompt':b['prompt'],'score':b['quality_score'],'quality_score':b['quality_score'],'align_score':b['align_score']}
        _,losses=compute_losses(target,down,'cpu',c)
        self.assertEqual(losses['quality_text_ce'].item(),0.)
        self.assertEqual(losses['objective_quality_text'].item(),0.)
        losses['loss'].backward()

    def test_shared_dual_real_pcgrad_training_and_downstream_adapter_order(self):
        from train_cm_pretrain import train_one_epoch, evaluate, build_optimizer
        from test_quality_route_ablations import Bank
        c=residual_config();c['model']['shared_simsiam_projection']=True
        # Avoid fake identical text BN gradients dominating this tiny fixture.
        c['train']['grad_clip']=0.
        source=make(c);source.iqa.quality_text_bank=Bank();b=batch();opt=build_optimizer(c,source)
        params=[source.iqa.shared_image_projector.net[0].weight,source.iqa.shared_text_projector.net[0].weight,
                source.iqa.quality_adapter.net[0].weight]
        before=[p.detach().clone() for p in params]
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(source,[b],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,1)
            metrics=evaluate(source,[b],torch.device('cpu'),c)
        for old,new in zip(before,params):self.assertFalse(torch.equal(old,new))
        self.assertTrue(torch.isfinite(torch.tensor(metrics['loss'])))
        target=make(c,joint=False);seen={}
        hooks=[getattr(target,n).register_forward_pre_hook(lambda m,args,n=n:seen.update({n:args[0]})) for n in ['align_global_adapter','align_patch_adapter','align_text_adapter']]
        out=target.forward_single(b['image_clean'],b['prompt'])
        for h in hooks:h.remove()
        g=target.backbone.fusion.semantic_proj(out['f_sem']);tokens=target._make_patch_tokens(out['f_loc'])
        torch.testing.assert_close(seen['align_global_adapter'],target.project_shared_image(g))
        torch.testing.assert_close(seen['align_patch_adapter'],target.project_shared_image(tokens))
        torch.testing.assert_close(seen['align_text_adapter'],torch.nn.functional.normalize(target.shared_text_projector(out['text_fused']),dim=-1))
        state=_extract_representation_state(source.state_dict());bad=dict(state);bad.pop('iqa.shared_text_projector.net.0.weight')
        with patch('torch.load',return_value={'model':bad}),self.assertRaisesRegex(RuntimeError,'Incomplete shared'):
            load_pretrained_backbone(target,'mock.pth',load_scope='representation')

    def test_eight_effective_configs_are_independent_and_keep_matched_residual_baseline(self):
        import train_cm_pretrain as pre
        import train_cm_iqa as down
        class Captured(Exception):pass
        for mode in ['noce','shared_pi_pt']:
            for stage in ['pretrain','agiqa1k','agiqa3k','aigciqa2023']:
                c=load_yaml(str(ROOT/f'configs/dual_residual_ablations/{stage}_{mode}.yaml'))
                self.assertEqual(c['model'].get('shared_simsiam_projection',False),mode=='shared_pi_pt')
                command=subprocess.check_output(['bash',str(ROOT/'scripts/run_dual_residual_ablation.sh'),mode,stage,'--dry-run'],text=True)
                if mode=='noce':self.assertNotIn('--quality-text-blend',command)
                parts=shlex.split(command);idx=next(i for i,v in enumerate(parts) if v in ['train_cm_pretrain.py','train_cm_iqa.py'])
                argv=parts[idx:];module=pre if stage=='pretrain' else down;seen=[]
                def snapshot(c,*args,**kwargs):seen.append(copy.deepcopy(c));raise Captured()
                with tempfile.TemporaryDirectory() as d:
                    argv[argv.index('--output-dir')+1]=d;argv[argv.index('--results-csv')+1]=d+'/r.csv'
                    with patch.object(sys,'argv',argv),patch.object(module,'dump_config_snapshot',side_effect=snapshot),contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(Captured):module.main()
                c=seen[0]
                self.assertEqual(c['model']['quality_adapter']['mode'],'residual_feature')
                self.assertEqual(c['model']['quality_global_feature'],'g')
                self.assertEqual(c['model']['local_encoder']['feature_stage'],'layer3')
                self.assertEqual(c['model']['quality_text_bank']['enabled'],mode!='noce')
                self.assertEqual(c['train']['lambda_quality_text_ce'],0. if mode=='noce' else .005)
                self.assertFalse(c['train'].get('reinit_local_projection',False))

if __name__=='__main__':unittest.main()
