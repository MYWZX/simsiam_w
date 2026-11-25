"""Numerical, transfer, forward/gradient and fixed-protocol evidence checks (CPU)."""
import contextlib
import copy
import io
import math
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from models.nss import LegacyKurtosisDescriptor,validate_nss
from models.alignment_evidence import AlignmentEvidence,spatial_diagnostics,token_coverage,lexical_mask
from models.two_stage_models import load_pretrained_backbone
from models.region_word_alignment import RegionWordAlignment
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import build_optimizer
from scripts import run_nss_kurtosis as kstudy,run_align_evidence as study
from test_projection_only import config as pre_config
from test_projection_ablations import make
from test_align_representation import model as original_model,config
import test_learnable_text_context as context_fixture


def model(c):
    # Match native image/text embedding widths in the tiny test CLIP pair.
    original=context_fixture.LearnableTextContextTests.make_encoder
    def encoder(self,n):
        text=original(self,n)
        text.model.text_projection=torch.nn.Parameter(torch.cat([text.model.text_projection]*2,dim=1),requires_grad=False)
        text.output_dim=8
        return text
    with patch.object(context_fixture.LearnableTextContextTests,'make_encoder',encoder):
        return original_model(c)



class KurtosisTests(unittest.TestCase):
    def test_first_five_exact_and_sixth_central_moment_reference(self):
        m=make(pre_config(True)).iqa;x=torch.rand(2,3,32,32)
        old=m._compute_nss_features(x,(2,2));rgb=m._normalize_img01(x)
        new=LegacyKurtosisDescriptor()(rgb,(2,2))
        torch.testing.assert_close(old[...,:5],new[...,:5],atol=0,rtol=0)
        gray=.299*rgb[:,:1]+.587*rgb[:,1:2]+.114*rgb[:,2:3]
        mu=F.avg_pool2d(gray,7,1,3)
        z=(gray-mu)/((F.avg_pool2d((gray-mu)**2,7,1,3)+1e-6).sqrt()+1e-6)
        patch=z[0,0,:16,:16].double();d=patch-patch.mean()
        expected=d.pow(4).mean()/d.square().mean().square()-3
        self.assertAlmostEqual(new[0,0,5].item(),expected.item(),places=4)
        self.assertFalse(torch.equal(new[...,5],old[...,5]))
        self.assertEqual(validate_nss({'descriptor':'legacy_kurt6_v1'}),('legacy_kurt6_v1',6))

    def test_black_flat_nondividing_grid_autocast_are_finite(self):
        d=LegacyKurtosisDescriptor()
        for grid in [(2,2),(3,5)]:
            with torch.autocast('cpu',dtype=torch.bfloat16):n=d(torch.zeros(2,3,17,19),grid)
            self.assertEqual(n.dtype,torch.float32)
            self.assertTrue(torch.isfinite(n).all());self.assertEqual(n[...,5].abs().sum(),0)

    def test_transfer_signature_prevents_silent_old_weight_reuse(self):
        c=pre_config(True);c['model'].setdefault('nss_text_gate',{}).update(descriptor='legacy_kurt6_v1',nss_dim=6)
        old=make(pre_config(True));new=make(c)
        source=_extract_representation_state(new.state_dict())
        with contextlib.redirect_stdout(io.StringIO()),patch('torch.load',return_value={'model':source}):
            load_pretrained_backbone(make(c).iqa,'fake',load_scope='representation')
            with self.assertRaisesRegex(RuntimeError,'NSS'):load_pretrained_backbone(old.iqa,'fake',load_scope='representation')
        source=_extract_representation_state(old.state_dict())
        with patch('torch.load',return_value={'model':source}),self.assertRaisesRegex(RuntimeError,'NSS'):
            load_pretrained_backbone(new.iqa,'fake',load_scope='representation')


class EvidenceTests(unittest.TestCase):
    def cfg(self,variant='diag'):
        c=config();e=study.make_runner().config(variant,'agiqa3k')['model']
        c['model']['align_evidence']=e['align_evidence']
        if 'align_region_word' in e:c['model']['align_region_word']=e['align_region_word']
        # Synthetic tokenizer deliberately has no decode; test lexical filtering
        # separately and disable only for tiny end-to-end fixtures.
        if 'coverage' in c['model']['align_evidence']:
            c['model']['align_evidence']['coverage']['lexical_mask']=False
        return c

    def test_diagnostic_only_exact_outputs_rng_and_parameters(self):
        torch.manual_seed(9);old=model(config()).eval();state=torch.random.get_rng_state()
        torch.manual_seed(9);new=model(self.cfg()).eval()
        torch.testing.assert_close(state,torch.random.get_rng_state(),atol=0,rtol=0)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
        image=torch.rand(3,3,16,16);text=['a','b','c']
        a=old.forward_single(image,text);b=new.forward_single(image,text)
        for key in ('s_quality','s_align','s_align_global','s_align_patch'):
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        self.assertIn('spatial_entropy',b['alignment_diagnostics'])
        self.assertEqual(b['alignment_diagnostics']['prompt_truncated'].shape,(3,))

    def test_temperature_bounds_gradients_and_attention_sharpening(self):
        e=AlignmentEvidence({'version':'align_evidence_v1','temperature':{'mode':'learned','initial':1.}})
        logits=torch.tensor([[.2,.8,-.6]])
        entropy=[]
        for value in [-7.,0.,7.]:
            with torch.no_grad():e.temperature_logit.fill_(value)
            t=e.temperature();self.assertTrue(.03<t<2.)
            p=(logits/t).softmax(-1);entropy.append(spatial_diagnostics(p)['spatial_entropy'].item())
        self.assertLess(entropy[0],entropy[-1])
        (logits/e.temperature()).softmax(-1)[0,0].backward()
        self.assertTrue(torch.isfinite(e.temperature_logit.grad));self.assertNotEqual(e.temperature_logit.grad.item(),0)

    def test_coverage_favors_missing_content_and_masks_punctuation(self):
        image=torch.tensor([[[1.,0.],[0.,1.]]]);text=torch.tensor([[[1.,0.],[-1.,0.],[0.,1.]]])
        weights=torch.tensor([[[1.,1.,0.],[0.,0.,0.]]]);mask=torch.tensor([[True,True,False]])
        e,t,match,importance=token_coverage(image,text,weights,mask,{'mode':'softmin','temperature':.1})
        self.assertGreater(importance[0,1],importance[0,0]);self.assertEqual(importance[0,2],0)
        torch.testing.assert_close(importance.sum(-1),torch.ones(1))
        class Tokenizer:
            def decode(self,ids):return {1:'cat',2:',',3:'!'}[ids[0]]
        ids=torch.tensor([[1,2,3],[2,3,3]]);valid=torch.ones_like(ids,dtype=torch.bool)
        keep,fallback=lexical_mask(ids,valid,Tokenizer())
        self.assertEqual(keep[0].tolist(),[True,False,False]);self.assertTrue(fallback[1]);self.assertTrue(keep[1].all())

    def test_region_temperature_coverage_and_context_receive_gradients(self):
        for variant in ('temperature','tokenmean','coverage','reference'):
            torch.manual_seed(8);m=model(self.cfg(variant)).eval()
            out=m.forward_single(torch.rand(3,3,16,16),['a','b','c'])
            out['s_align'].sum().backward()
            p=m.align_evidence.temperature_logit
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad));self.assertGreater(p.grad.abs().sum(),0)
            self.assertGreater(m.backbone.text_encoder.context_vectors.grad.abs().sum(),0)
            self.assertTrue(all(p.grad is None for p in m.backbone.text_encoder.model.parameters()))
            if variant=='reference':
                for p in m.align_evidence.parameters():self.assertIsNotNone(p.grad)
            cfg=self.cfg(variant);cfg['train']=study.make_runner().config(variant,'agiqa3k')['train']
            opt=build_optimizer(cfg,m);owner={id(p):g for g in opt.param_groups for p in g['params']}
            for p in m.align_evidence.parameters():self.assertEqual(owner[id(p)]['name'],'align');self.assertEqual(owner[id(p)]['lr'],1e-4)
            self.assertEqual(owner[id(m.backbone.fusion.semantic_proj.weight)]['lr'],1e-5)

    def test_quality_and_global_align_identical_for_all_nonreference_changes(self):
        image=torch.rand(3,3,16,16);text=['a','b','c'];torch.manual_seed(10);baseline=model(config()).eval()
        out=baseline.forward_single(image,text)
        for variant in study.VARIANTS:
            torch.manual_seed(10);m=model(self.cfg(variant)).eval();new=m.forward_single(image,text)
            for key in ('s_quality','s_quality_global','s_quality_patch','s_align_global'):
                torch.testing.assert_close(out[key],new[key],atol=0,rtol=0)

    def test_native_reference_formula_not_256_space_or_detached(self):
        e=AlignmentEvidence({'version':'align_evidence_v1','native_reference':{'enabled':True,'initial_mix':.05}})
        image=torch.randn(2,8,requires_grad=True);text=torch.randn(2,8,requires_grad=True);base=torch.tensor([.3,.7])
        y,d=e.reference(image,text,base)
        torch.testing.assert_close(y,.95*base+.05*torch.sigmoid(F.cosine_similarity(image,text)))
        y.sum().backward();self.assertGreater(image.grad.abs().sum(),0);self.assertGreater(text.grad.abs().sum(),0)
        with self.assertRaises(ValueError):e.reference(image,torch.randn(2,4),base)

    def test_evaluation_aggregates_only_valid_alignment_labels(self):
        from train_cm_iqa import evaluate
        c=self.cfg();m=model(c).eval()
        x=torch.rand(3,3,16,16)
        b={'image':x,'prompt':['a','b','c'],'score':torch.tensor([.2,.4,.7]),
           'quality_score':torch.tensor([.2,.4,.7]),'align_score':torch.tensor([.3,.5,.8]),
           'align_mask':torch.tensor([1.,0.,1.])}
        out=m.forward_single(x,b['prompt'])
        metrics=evaluate(m,[b],torch.device('cpu'),c)
        expected=out['alignment_diagnostics']['spatial_max'][[0,2]].mean().item()
        self.assertAlmostEqual(metrics['val_evidence_spatial_max'],expected,places=6)
        self.assertIn('val_evidence_s_align_MAE',metrics)
        self.assertEqual(metrics['val_evidence_native_N'],2)

    def test_invalid_config_rejected(self):
        for cfg in ({'version':'bad'}, {'version':'align_evidence_v1','typo':True},
                    {'version':'align_evidence_v1','temperature':{'initial':0}},
                    {'version':'align_evidence_v1','coverage':{'mode':'softmin'}},
                    {'version':'align_evidence_v1','native_reference':{'initial_mix':1}}):
            with self.assertRaises(ValueError):AlignmentEvidence(cfg)
        c=self.cfg();c['train']['use_joint_iqa_pretraining']=True
        with self.assertRaises(ValueError):model(c)
        c=self.cfg();c['model']['align_local_temperature']=.5
        with self.assertRaises(ValueError):model(c)


class ProtocolTests(unittest.TestCase):
    def test_independent_paths_sources_fixed_protocol_and_order(self):
        seen=set()
        for nss in ('legacy','kurt6'):
            r=study.make_runner(nss)
            for ds in r.DATASETS:
                for variant in study.VARIANTS:
                    c=r.config(variant,ds);self.assertNotIn(c['output_dir'],seen);seen.add(c['output_dir'])
                    self.assertEqual(c['seed'],42);self.assertEqual(c['split_seed'],42)
                    self.assertEqual(c['train']['epochs'],60);self.assertTrue(c['train']['compare_selection_policies'])
                    self.assertEqual(c['train']['layerwise_lr']['shared_projection_mult'],1.)
                    self.assertEqual(c['model']['nss_text_gate'].get('descriptor','legacy'),'legacy' if nss=='legacy' else 'legacy_kurt6_v1')
        for stage in ('pretrain','agiqa1k','agiqa3k','aigciqa2023'):
            c=kstudy.expected_config('kurt6',stage)
            if stage=='pretrain':
                self.assertNotIn('pretrained_checkpoint',c['train']);self.assertNotIn('resume',c['train'])
            else:self.assertIn('legacy_kurt6_v1',c['train']['pretrained_checkpoint'])

class ArtifactTests(unittest.TestCase):
    def test_plot_script_renders_all_panels_from_logged_curves(self):
        import tempfile
        from pathlib import Path
        import csv,json
        from scripts import plot_align_evidence as plotting
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            class Runner:
                def paths(self,v,ds):return root,root/(v+'.csv')
            keys=('val_val_align_SROCC','val_val_quality_SROCC','val_val_evidence_spatial_entropy',
                  'val_val_evidence_spatial_max','val_val_evidence_temperature','val_val_evidence_s_align_MAE')
            for v in study.VARIANTS:
                with (root/(v+'.csv')).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                    for ep in (1,2):w.writerow({'event':'epoch','epoch':ep,'metrics_json':json.dumps({k:.1*ep for k in keys})})
            with patch.object(plotting,'make_runner',return_value=Runner()):plotting.render('legacy','agiqa3k',root/'figure')
            self.assertGreater((root/'figure.png').stat().st_size,10000)
            self.assertTrue((root/'figure.pdf').read_bytes().startswith(b'%PDF'))

    def test_runner_refuses_cpu_without_output_creation(self):
        import tempfile
        from pathlib import Path
        r=study.make_runner('legacy')
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'source',return_value={}),patch.object(r,'control'),patch('torch.cuda.is_available',return_value=False):
            r.ROOT=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('diag','agiqa3k')
            self.assertFalse((Path(tmp)/'outputs').exists())

if __name__=='__main__':unittest.main()
