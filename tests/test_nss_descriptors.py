"""Analytical numerical tests and tiny fake-backbone forwards; no real training."""
import contextlib
import copy
import io
import math
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from models.nss import StandardMSCNDescriptor,validate_nss
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state,build_optimizer as pre_optim
from train_cm_iqa import build_optimizer as down_optim
from test_projection_only import config
from test_projection_ablations import make
from test_dual_quality_representation import batch,views


class NSSDescriptorTests(unittest.TestCase):
    def setUp(self):
        rng=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,rng)
        torch.manual_seed(42)

    def cfg(self,mode):
        c=config(True)
        c['model'].setdefault('nss_text_gate',{}).update(descriptor=mode,nss_dim=2 if mode=='mscn_ggd_v1' else 6)
        return c

    def test_gaussian_mscn_matches_independent_double_reference(self):
        rgb=torch.rand(2,3,32,32)
        d=StandardMSCNDescriptor('mscn_stats6_v1');gray,z=d.mscn(rgb)
        x=torch.arange(-3,4,dtype=torch.float64)
        g=torch.exp(-.5*(x/(7/6)).square());g/=g.sum();k=(g[:,None]*g[None,:])[None,None]
        ref=(rgb.double()*torch.tensor([.299,.587,.114],dtype=torch.float64)[None,:,None,None]).sum(1,keepdim=True)
        blur=lambda a:F.conv2d(F.pad(a,(3,3,3,3),mode='replicate'),k)
        mu=blur(ref);var=(blur(ref.square())-mu.square()).clamp_min(0)
        torch.testing.assert_close(z.double(),(ref-mu)/(var.sqrt()+1/255),atol=3e-5,rtol=3e-5)
        self.assertAlmostEqual(d.kernel.sum().item(),1,places=6)

    def test_six_statistics_match_patchwise_central_moments(self):
        d=StandardMSCNDescriptor('mscn_stats6_v1');rgb=torch.rand(1,3,32,32)
        gray,z=d.mscn(rgb);a=z[0,0,:16,:16].double();mean=a.mean();std=((a-mean).square().mean()).sqrt()
        dx=F.pad(gray[:,:,:,1:]-gray[:,:,:,:-1],(0,1,0,0));dy=F.pad(gray[:,:,1:,:]-gray[:,:,:-1,:],(0,0,0,1))
        grad=((dx.square()+dy.square()+1e-12).sqrt()-1e-6)[0,0,:16,:16].mean()
        expected=torch.stack([mean,std,a.abs().mean(),((a-mean)/std).pow(3).mean(),((a-mean)/std).pow(4).mean()-3,grad.double()])
        out=d(rgb,(2,2));torch.testing.assert_close(out[0,0].double(),expected,atol=2e-5,rtol=2e-5)
        self.assertEqual(out.shape,(1,4,6));self.assertFalse(torch.equal(out[:,:,1],out[:,:,5]))

    def test_ggd_recovers_known_shapes_and_variance(self):
        d=StandardMSCNDescriptor('mscn_ggd_v1')
        alpha=torch.tensor([.5,1.,2.,5.],dtype=torch.float64)
        ratio=torch.exp(2*torch.lgamma(2/alpha)-torch.lgamma(1/alpha)-torch.lgamma(3/alpha))
        variance=torch.tensor([.2,1.,2.,4.]);absolute=(ratio.float()*variance).sqrt()
        fit,var=d.fit_ggd_moments(absolute,variance)
        torch.testing.assert_close(fit,alpha.float(),atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(var,variance)
        fit,_=d.fit_ggd_moments(torch.tensor([0.,100.]),torch.tensor([0.,1.]))
        torch.testing.assert_close(fit,torch.tensor([2.,10.]))

    def test_finite_constant_tiny_low_contrast_amp_and_gradients(self):
        for mode in ('mscn_stats6_v1','mscn_ggd_v1'):
            d=StandardMSCNDescriptor(mode)
            for rgb in (torch.zeros(1,3,8,8),torch.ones(1,3,8,8)*.5,torch.rand(1,3,8,8)*1e-5+.5,torch.rand(1,3,32,32)):
                rgb.requires_grad_();out=d(rgb,(2,2));self.assertTrue(torch.isfinite(out).all())
                out.sum().backward();self.assertTrue(torch.isfinite(rgb.grad).all())
                with torch.autocast('cpu',dtype=torch.bfloat16):amp=d(rgb,(2,2))
                torch.testing.assert_close(out,amp)
            if mode=='mscn_ggd_v1':
                out=d(torch.zeros(1,3,8,8),(2,2));torch.testing.assert_close(out[0,:,0],torch.full((4,),2.));self.assertEqual(out[0,:,1].sum(),0)

    def test_invalid_modes_and_dimensions(self):
        for cfg in ({'descriptor':'unknown'},{'descriptor':'mscn_ggd_v1','nss_dim':6},{'descriptor':'legacy','nss_dim':2}):
            with self.assertRaises(ValueError):validate_nss(cfg)
        c=self.cfg('mscn_stats6_v1');c['model']['fusion']['mode']='legacy'
        with self.assertRaisesRegex(ValueError,'projection_only'):make(c)

    def test_both_forward_backward_and_crop_alignment(self):
        for mode in ('mscn_stats6_v1','mscn_ggd_v1'):
            c=self.cfg(mode);m=make(c);b=batch();out=views(m,b)
            self.assertTrue(torch.isfinite(out['quality_pred_clean']).all())
            out['quality_pred_clean'].sum().backward()
            for name in ('nss_proj.0.weight','nss_gate.0.weight'):
                p=dict(m.iqa.named_parameters())[name];self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)
            crops=torch.randn(2,3,3,16,16);f_loc=torch.empty(2,8,6,2)
            got=m.iqa._compute_local_nss_features(torch.randn(2,3,16,16),f_loc,crops)
            expected=m.iqa._compute_nss_features(crops.reshape(6,3,16,16),(2,2)).reshape(2,-1,m.iqa.nss_dim)
            torch.testing.assert_close(got,expected)
            self.assertEqual(got.shape,(2,12,m.iqa.nss_dim))

    def test_strict_transfer_rejects_missing_cross_mode_and_legacy(self):
        modes=('legacy','mscn_stats6_v1','mscn_ggd_v1')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(tmp)/'source.pth'
            for source in modes:
                m=make(self.cfg(source));state=_extract_representation_state(m.state_dict())
                torch.save({'model':state},p)
                for target in modes:
                    t=make(self.cfg(target))
                    if source==target:
                        load_pretrained_backbone(t.iqa,p,load_scope='representation')
                        for k,v in state.items():
                            torch.testing.assert_close(t.state_dict()[k],v,atol=0,rtol=0)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'NSS'):load_pretrained_backbone(t.iqa,p,load_scope='representation')
                if source!='legacy':
                    broken=dict(state);del broken['iqa.nss_gate.0.weight'];torch.save({'model':broken},p)
                    with self.assertRaisesRegex(RuntimeError,'NSS'):load_pretrained_backbone(make(self.cfg(source)).iqa,p,load_scope='representation')

    def test_default_legacy_state_output_and_optimizer_ownership(self):
        c=config(True);torch.manual_seed(42);old=make(c)
        c2=copy.deepcopy(c);c2['model'].setdefault('nss_text_gate',{})['descriptor']='legacy';torch.manual_seed(42);explicit=make(c2)
        self.assertEqual(old.state_dict().keys(),explicit.state_dict().keys())
        for k,v in old.state_dict().items():torch.testing.assert_close(v,explicit.state_dict()[k],atol=0,rtol=0)
        b=batch();a=views(old,b);z=views(explicit,b)
        torch.testing.assert_close(a['quality_pred_clean'],z['quality_pred_clean'],atol=0,rtol=0)
        for optimizer,joint in ((pre_optim,True),(down_optim,False)):
            ownership=[]
            for mode in ('legacy','mscn_stats6_v1','mscn_ggd_v1'):
                cc=self.cfg(mode);cc['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'fusion_mult':.1,'shared_projection_mult':1.})
                mm=make(cc,joint=joint);oo=optimizer(cc,mm);names={id(p):n for n,p in mm.named_parameters()}
                ownership.append({names[id(p)]:(g.get('name'),g['lr']) for g in oo.param_groups for p in g['params']})
            self.assertEqual(ownership[0],ownership[1]);self.assertEqual(ownership[0],ownership[2])

if __name__=='__main__':unittest.main()
