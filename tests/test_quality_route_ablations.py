"""Quality g-vs-z dependency isolation and strict no-local execution."""
import copy
from unittest.mock import patch
import unittest
import torch
from torch import nn
from models.two_stage_models import CrossModalIQAFullModel
from train_cm_iqa import evaluate, parse_args, compute_losses

class Global(nn.Module):
    def __init__(self,**kwargs):super().__init__();self.proj=nn.Linear(3,8)
    def forward(self,x):return self.proj(x.mean((2,3)))
class Local(nn.Module):
    output_dim=4
    def __init__(self,**kwargs):super().__init__();self.proj=nn.Conv2d(3,4,1)
    def forward(self,x):return self.proj(torch.nn.functional.adaptive_avg_pool2d(x,(2,2)))
class Text(nn.Module):
    output_dim=6
    def __init__(self,**kwargs):super().__init__()
    def forward(self,prompts,device):return torch.ones(len(prompts),6,device=device)

class Bank(nn.Module):
    def __init__(self):
        super().__init__();self.image_to_text=nn.Linear(8,5);self.register_buffer('centers',torch.linspace(0,1,5))
    def context_anchor_loss(self):return self.centers.new_zeros(())
    def forward(self,x):
        self.seen=x
        logits=self.image_to_text(x)
        return {'quality_text_logits':logits,'quality_text_score':logits.softmax(-1)@self.centers}

def config(feature='z',no_local=False):
    return {'dataset':{'input_size':16},'model':{
        'global_encoder':{'pretrained':False,'output_dim':8},'local_encoder':{'pretrained':False,'output_dim':4},
        'fusion':{'embed_dim':8,'num_heads':2,'dropout':0.},'heads':{'mlp_hidden_dim':16,'dropout':0.},
        'quality_global_feature':feature,'quality_no_local_branch':no_local,'quality_branch_use_text_gate':False,
        'quality_adapter':{'enabled':True,'hidden_dim':16,'output_dim':8,'dropout':0.,'last_bn':True,'mode':'residual_feature','residual_init':.1}},
        'train':{'two_output_only':True,'lambda_quality':1.,'lambda_align':1.,'lambda_quality_text_ce':.005,'lambda_neg':0.}}

def build(feature='z',no_local=False):
    cfg=config(feature,no_local)
    with patch('models.two_stage_models.GlobalVisualEncoder',Global),patch('models.two_stage_models.ResNet50LocalEncoder',Local),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text):
        return CrossModalIQAFullModel(cfg).eval(),cfg

def grads(loss,params):return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)

class QualityRouteTests(unittest.TestCase):
    def test_g_replaces_adapter_and_entire_residual_input_but_preserves_align_and_local(self):
        torch.manual_seed(42);base,_=build()
        torch.manual_seed(42);model,_=build('g')
        for key,v in base.state_dict().items():torch.testing.assert_close(v,model.state_dict()[key],rtol=0,atol=0)
        captured=[];h=model.quality_adapter.register_forward_pre_hook(lambda m,ins:captured.append(ins[0]))
        x=torch.randn(3,3,16,16);a=base.forward_single(x,['cat']*3);b=model.forward_single(x,['cat']*3);h.remove()
        expected=model.backbone.fusion.semantic_proj(model.backbone.global_encoder(x))
        torch.testing.assert_close(captured[0],expected)
        r,_=model._quality_residual_feature(expected,b['z_quality_global'])
        torch.testing.assert_close(b['s_quality_global'],torch.sigmoid(model.global_quality_head(r).squeeze(-1)))
        torch.testing.assert_close(a['s_align'],b['s_align'])
        torch.testing.assert_close(a['s_quality_patch'],b['s_quality_patch'])
        torch.testing.assert_close(a['z'],b['z'])
        self.assertFalse(torch.allclose(a['s_quality_global'],b['s_quality_global']))
        self.assertTrue(b['quality_local_enabled'])
        # Total Quality still reaches ResNet through local scoring, but not
        # Fusion attention/FFN/norms; its existing semantic projection remains trainable.
        params=[p for n,p in model.backbone.fusion.named_parameters() if not n.startswith('semantic_proj')]
        self.assertTrue(all(g is None or g.abs().sum()==0 for g in grads(b['s_quality'].sum(),params)))
        g=grads(b['s_quality_global'].sum(),[model.backbone.fusion.semantic_proj.weight])[0]
        self.assertGreater(g.abs().sum().item(),0.)
        g=grads(b['s_quality'].sum(),[model.backbone.local_encoder.proj.weight])[0]
        self.assertGreater(g.abs().sum().item(),0.)

    def test_strict_no_local_skips_computation_and_keeps_cross_attention(self):
        torch.manual_seed(42);base,_=build()
        torch.manual_seed(42);model,cfg=build(no_local=True)
        for key,v in base.state_dict().items():torch.testing.assert_close(v,model.state_dict()[key],rtol=0,atol=0)
        seen=[]
        hooks=[getattr(model,n).register_forward_hook(lambda m,i,o,n=n:seen.append(n)) for n in ['nss_proj','nss_gate','patch_quality_head','local_patch_proj']]
        with patch.object(model,'_compute_local_nss_features',side_effect=AssertionError('NSS must not execute')):
            x=torch.randn(3,3,16,16);a=base.forward_single(x,['cat']*3);b=model.forward_single(x,['cat']*3)
        for h in hooks:h.remove()
        self.assertEqual(seen,['local_patch_proj']) # only Align's shared projection call
        self.assertIsNone(b['s_quality_patch']);self.assertIsNone(b['quality_patch_weights']);self.assertIsNone(b['patch_tokens'])
        self.assertFalse(b['quality_local_enabled']);self.assertEqual(b['quality_mix'].item(),0.)
        torch.testing.assert_close(b['s_quality'],b['s_quality_global'])
        torch.testing.assert_close(a['s_quality_global'],b['s_quality_global'])
        torch.testing.assert_close(a['s_align'],b['s_align'])
        g=grads(b['s_quality'].sum(),[model.backbone.fusion.attn.in_proj_weight,model.backbone.local_encoder.proj.weight])
        self.assertTrue(all(t.abs().sum()>0 for t in g))
        for module in [model.nss_proj,model.nss_gate,model.text_gate,model.patch_quality_head]:self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.local_patch_proj.parameters()))
        batch={'image':x,'prompt':['cat']*3,'score':torch.tensor([.2,.5,.8]),'quality_score':torch.tensor([.2,.5,.8]),'align_score':torch.tensor([.3,.6,.7])}
        m=evaluate(model,[batch],'cpu',cfg)
        self.assertEqual(m['val_quality_local_enabled'],0.)
        self.assertNotIn('val_quality_patch_PLCC',m);self.assertNotIn('val_quality_branch_error_PLCC',m)
        self.assertIn('val_quality_global_PLCC',m)

    def test_quality_text_ce_remains_trainable_in_both_routes(self):
        for feature,no_local in [('g',False),('z',True)]:
            model,cfg=build(feature,no_local);model.quality_text_bank=Bank()
            batch={'image':torch.randn(3,3,16,16),'prompt':['cat']*3,'score':torch.tensor([.2,.5,.8]),'quality_score':torch.tensor([.2,.5,.8]),'align_score':torch.tensor([.3,.6,.7])}
            out,losses=compute_losses(model,batch,'cpu',cfg)
            torch.testing.assert_close(model.quality_text_bank.seen,out['z_quality_global'])
            losses['loss'].backward()
            self.assertGreater(model.quality_text_bank.image_to_text.weight.grad.abs().sum().item(),0.)
            self.assertGreater(model.quality_adapter.net[0].weight.grad.abs().sum().item(),0.)

    def test_cli_and_configuration_validation(self):
        with patch('sys.argv',['train_cm_iqa.py','--config','unused.yaml','--quality-global-feature','g','--quality-no-local-branch']):
            args=parse_args();self.assertEqual(args.quality_global_feature,'g');self.assertTrue(args.quality_no_local_branch)
        for cfg in [config('bad'),config('g'),config(no_local=True)]:
            if cfg['model']['quality_global_feature']=='g':cfg['model']['quality_local_attention']={'enabled':True,'num_heads':2}
            elif cfg['model']['quality_no_local_branch']:cfg['model']['quality_auxiliary_heads']={'enabled':True,'components':['artifact']}
            with patch('models.two_stage_models.GlobalVisualEncoder',Global),patch('models.two_stage_models.ResNet50LocalEncoder',Local),patch('models.two_stage_models.FrozenCLIPTextEncoder',Text):
                with self.assertRaises(ValueError):CrossModalIQAFullModel(cfg)

if __name__=='__main__':unittest.main()
