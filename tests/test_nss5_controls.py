"""Five-stat numerical equivalence, matched initialization, routes and transfer."""
import contextlib,copy,io,tempfile,unittest
from pathlib import Path
import torch
from models.nss import FiveStatDescriptor,collapse_legacy_nss_state,validate_nss
from models.two_stage_models import load_pretrained_backbone
from train_cm_pretrain import _extract_representation_state,build_optimizer as pre_optimizer
from train_cm_iqa import build_optimizer as down_optimizer
from test_projection_only import config
from test_projection_ablations import make
from test_dual_quality_representation import batch,views
from scripts import run_nss5_controls as sweep
from scripts.migrate_nss5_checkpoint import migrate

class FiveStatTests(unittest.TestCase):
    def setUp(self):
        state=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,state);torch.manual_seed(42)

    def cfg(self,variant):
        c=config(True);mode,path=sweep.VARIANTS[variant]
        c['model'].setdefault('nss_text_gate',{}).update(descriptor=mode,nss_dim=5,pathway=path)
        return c

    def test_exact_first_five_legacy_channels_and_common_statistics_formulas(self):
        old=make(config(True));image=torch.randn(2,3,32,32);rgb=old.iqa._normalize_img01(image)
        old6=old.iqa._compute_nss_features(image,(2,2));a=FiveStatDescriptor('legacy_stats5_v1')(rgb,(2,2));b=FiveStatDescriptor('mscn_stats5_v1')(rgb,(2,2))
        torch.testing.assert_close(a,old6[:,:,:5],atol=0,rtol=0)
        torch.testing.assert_close(old6[:,:,1],old6[:,:,5],atol=0,rtol=0)
        # Normalization comparison leaves raw luminance gradient channel identical.
        torch.testing.assert_close(a[:,:,3],b[:,:,3],atol=0,rtol=0)
        for n in (a,b):torch.testing.assert_close(n[:,:,4],n[:,:,2]/(torch.sqrt(n[:,:,1]+1e-6)+1e-6))
        self.assertFalse(torch.equal(a[:,:,:3],b[:,:,:3]))

    def test_four_variants_match_all_common_initialization(self):
        models={}
        for v in sweep.VARIANTS:
            torch.manual_seed(42);models[v]=make(self.cfg(v))
        canonical=models['legacy_both'].state_dict();e=models['legacy_both'].iqa.backbone.fused_dim
        torch.manual_seed(42);old=make(config(True))
        folded=collapse_legacy_nss_state(old.state_dict(),e)
        for k,value in canonical.items():torch.testing.assert_close(value,folded[k],atol=0,rtol=0)
        for v,m in models.items():
            for k,value in m.state_dict().items():
                if k.endswith(('nss_descriptor_signature','nss_pathway_signature')):continue
                expected=canonical[k]
                if v=='legacy_inject' and k=='iqa.nss_gate.0.weight':expected=expected[:,:e]
                torch.testing.assert_close(value,expected,atol=0,rtol=0)
        self.assertFalse(any(k.startswith('iqa.nss_proj.') for k in models['legacy_gate'].state_dict()))

    def test_merged_old_weights_preserve_all_eval_outputs(self):
        old=make(config(True));new=make(self.cfg('legacy_both'));e=old.iqa.backbone.fused_dim
        state=old.state_dict();merged=collapse_legacy_nss_state(state,e)
        new.load_state_dict(merged,strict=True);b=batch();a,z=views(old,b),views(new,b)
        checked=[]
        for k,value in a.items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,z[k],atol=3e-6,rtol=3e-6,msg=k);checked.append(k)
        self.assertIn('quality_pred_clean',checked);self.assertIn('align_pred_clean',checked)
        self.assertGreater(len(checked),20)
        for k,value in state.items():
            if not k.endswith(('nss_proj.0.weight','nss_gate.0.weight')):torch.testing.assert_close(value,merged[k],atol=0,rtol=0)

    def test_pathways_execute_only_enabled_inputs_and_receive_gradients(self):
        for v,(_,path) in sweep.VARIANTS.items():
            m=make(self.cfg(v));seen=[];gate_inputs=[]
            h=m.iqa.nss_proj.register_forward_hook(lambda *args:seen.append(1))
            g=m.iqa.nss_gate.register_forward_pre_hook(lambda mod,args:gate_inputs.append(args[0]))
            out=views(m,batch());h.remove();g.remove();out['quality_pred_clean'].sum().backward()
            self.assertEqual(bool(seen),path!='gate_only')
            e=m.iqa.backbone.fused_dim;self.assertEqual(gate_inputs[0].shape[-1],e if path=='inject_only' else e+5)
            self.assertIsNotNone(m.iqa.nss_gate[0].weight.grad)
            self.assertGreater(m.iqa.nss_gate[0].weight.grad.abs().sum(),0)
            if path!='gate_only':self.assertGreater(m.iqa.nss_proj[0].weight.grad.abs().sum(),0)
            self.assertTrue(torch.isfinite(out['quality_pred_clean']).all())

    def test_transfer_requires_matching_descriptor_and_path(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(tmp)/'source.pth'
            for v in sweep.VARIANTS:
                source=make(self.cfg(v));state=_extract_representation_state(source.state_dict());torch.save({'model':state},p)
                for w in sweep.VARIANTS:
                    target=make(self.cfg(w))
                    if v!=w:
                        with self.assertRaisesRegex(RuntimeError,'NSS'):load_pretrained_backbone(target.iqa,p,load_scope='representation')
                    else:
                        load_pretrained_backbone(target.iqa,p,load_scope='representation')
                        for k,value in state.items():torch.testing.assert_close(target.state_dict()[k],value,atol=0,rtol=0)
                broken=dict(state);del broken['iqa.nss_pathway_signature'];torch.save({'model':broken},p)
                with self.assertRaisesRegex(RuntimeError,'NSS'):load_pretrained_backbone(make(self.cfg(v)).iqa,p,load_scope='representation')

    def test_optimizer_ownership_and_lr_remain_for_all_surviving_parameters(self):
        for optimizer,joint in ((pre_optimizer,True),(down_optimizer,False)):
            reference=None
            for v in sweep.VARIANTS:
                c=self.cfg(v);c['train'].update(lr=1e-4,layerwise_lr={'enabled':True,'fusion_mult':.1,'shared_projection_mult':1.})
                m=make(c,joint=joint);o=optimizer(c,m);names={id(p):n for n,p in m.named_parameters()}
                actual={names[id(p)]:(g.get('name'),g['lr']) for g in o.param_groups for p in g['params']}
                if reference is None:reference=actual
                self.assertEqual(actual,{k:v for k,v in reference.items() if k in actual})
                projection=next(v for k,v in actual.items() if k.endswith('fusion.semantic_proj.weight'))
                self.assertAlmostEqual(projection[1],1e-5)

    def test_constant_amp_and_multicrop_layout(self):
        for v in sweep.VARIANTS:
            m=make(self.cfg(v));crops=torch.randn(2,3,3,16,16)
            n=m.iqa._compute_local_nss_features(torch.zeros(2,3,16,16),torch.empty(2,8,6,2),crops)
            ref=m.iqa._compute_nss_features(crops.reshape(6,3,16,16),(2,2)).reshape(2,12,5)
            torch.testing.assert_close(n,ref)
            d=FiveStatDescriptor(sweep.VARIANTS[v][0]);flat=torch.zeros(1,3,8,8)
            with torch.autocast('cpu',dtype=torch.bfloat16):x=d(flat,(2,2))
            self.assertTrue(torch.isfinite(x).all());torch.testing.assert_close(x,d(flat,(2,2)))
        with self.assertRaises(ValueError):validate_nss({'descriptor':'legacy','pathway':'inject_only'})
        with self.assertRaises(ValueError):validate_nss({'descriptor':'legacy_stats5_v1','pathway':'unknown'})

    def test_actual_checkpoint_migration_tool_preserves_source_and_marks_verification_only(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            source=Path(tmp)/'old.pth';output=Path(tmp)/'new.pth';m=make(config(True))
            torch.save({'model':_extract_representation_state(m.state_dict()),'cfg':config(True),'epoch':5,'run_id':'old','optimizer':{}},source)
            before=source.read_bytes();rng=torch.random.get_rng_state()
            report=migrate(source,output);converted=torch.load(output,weights_only=False)
            self.assertEqual(source.read_bytes(),before);self.assertEqual(converted['checkpoint_scope'],'migration_verification_only')
            self.assertNotIn('optimizer',converted);self.assertIn('optimizer',report['dropped_training_state_fields'])
            torch.testing.assert_close(rng,torch.random.get_rng_state(),atol=0,rtol=0)
            with self.assertRaises(FileExistsError):migrate(source,output)
            with self.assertRaises(ValueError):collapse_legacy_nss_state(converted['model'],m.iqa.backbone.fused_dim)

if __name__=='__main__':unittest.main()
