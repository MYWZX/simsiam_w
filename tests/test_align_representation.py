"""Synthetic end-to-end route/transfer, real tiny OpenCLIP, gradient and protocol checks."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from open_clip.transformer import VisionTransformer
from models.backbones import GlobalVisualEncoder
from models.two_stage_models import CrossModalIQAFullModel, load_pretrained_backbone
from models.region_word_alignment import RegionWordAlignment, content_token_mask
from scripts import run_align_representation as study
from train_cm_pretrain import _extract_representation_state
from train_cm_iqa import build_optimizer, get_selection_value
from utils.selection_policies import configure_selection_comparison, SelectionComparison
from utils.task_gradient_probe import probe_task_gradients, maybe_record_probe, new_region_parameter_record
import test_learnable_text_context as ctx_fixture
from test_frozen_visual_encoder_mode import LocalBN
from test_dual_quality_residual import residual_config
import test_align_tuning as fixture


class TinyVisual(GlobalVisualEncoder):
    def __init__(self, **kw):
        nn.Module.__init__(self)
        self._uses_open_clip=True;self.source_normalization='none';self.normalize_output=True
        self.backbone=VisionTransformer(image_size=16,patch_size=8,width=8,layers=1,heads=2,mlp_ratio=2,output_dim=8)
        self.proj=nn.Identity();self.output_dim=8
        self.register_buffer('model_mean',torch.zeros(1,3,1,1),persistent=False)
        self.register_buffer('model_std',torch.ones(1,3,1,1),persistent=False)


def config():
    c=residual_config();c['model']['fusion']['mode']='projection_only_v1';c['model']['shared_projection_project_patches']=False;c['model']['shared_simsiam_projection']=True
    c['train'].update(use_joint_iqa_pretraining=False,two_output_only=True,quality_only=False,lambda_neg=0,lambda_hard_neg=0,lambda_align_rank=0)
    return c


def model(c):
    with patch('models.two_stage_models.GlobalVisualEncoder',TinyVisual),patch('models.two_stage_models.ResNet50LocalEncoder',LocalBN),patch('models.two_stage_models.FrozenCLIPTextEncoder',side_effect=lambda **kw:ctx_fixture.LearnableTextContextTests().make_encoder(5)):
        return CrossModalIQAFullModel(c)


class RouteTests(unittest.TestCase):
    def setUp(self):
        state=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,state);torch.manual_seed(42)

    def test_real_vit_same_pooled_output_single_call_and_restored_flag(self):
        m=TinyVisual().eval();x=torch.rand(3,3,16,16);a=m(x)
        with patch.object(m.backbone,'forward',wraps=m.backbone.forward) as call:b,tokens=m(x,return_tokens=True)
        self.assertEqual(call.call_count,1);self.assertFalse(m.backbone.output_tokens)
        self.assertEqual(tokens.shape,(3,4,8));torch.testing.assert_close(a,b,rtol=0,atol=0)
        with patch.object(m.backbone,'forward',side_effect=RuntimeError('fail')):
            with self.assertRaises(RuntimeError):m(x,return_tokens=True)
        self.assertFalse(m.backbone.output_tokens)

    def test_mask_excludes_prefix_eos_padding_preserves_real_zero_token(self):
        ids=torch.tensor([[1,0,0,0,0,0,0,7,99,0]])
        mask=content_token_mask(ids,5)
        self.assertEqual(mask.tolist(),[[False]*6+[True,True,False,False]])
        with self.assertRaises(ValueError):content_token_mask(torch.tensor([[1,0,0,99,0]]),5)

    def test_matching_formula_mask_and_context_gradient_with_frozen_text(self):
        text=ctx_fixture.LearnableTextContextTests().make_encoder(5)
        tokens,ids=text.encode_token_features(['a','b'],torch.device('cpu'))
        mask=content_token_mask(ids,5);regions=torch.randn(2,4,8,requires_grad=True)
        matcher=RegionWordAlignment(8,4,8);e,t,w=matcher(regions,tokens,mask)
        self.assertEqual(w.shape,(2,4,8));torch.testing.assert_close(w.sum(1),mask.float())
        self.assertEqual(w[~mask[:,None,:].expand_as(w)].abs().sum(),0)
        (e.square().sum()+t[:,0].sum()).backward()
        self.assertGreater(text.context_vectors.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in text.model.parameters()))
        with torch.no_grad():self.assertFalse(text.encode_token_features(['a'],torch.device('cpu'))[0].requires_grad)

    def test_quality_parity_transfer_gradients_rng_and_optimizer(self):
        c=config();torch.manual_seed(7);old=model(c).eval();after=torch.random.get_rng_state()
        cc=copy.deepcopy(c);cc['model']['align_region_word']={'version':'clip_region_word_v1'}
        torch.manual_seed(7);new=model(cc).eval();torch.testing.assert_close(torch.random.get_rng_state(),after,rtol=0,atol=0)
        inherited={k:v for k,v in new.state_dict().items() if not k.startswith('align_region_word.')}
        self.assertEqual(set(inherited),set(old.state_dict()))
        for k,v in inherited.items():torch.testing.assert_close(v,old.state_dict()[k],rtol=0,atol=0)
        with torch.no_grad():old.backbone.fusion.semantic_proj.weight.add_(.17)
        source=_extract_representation_state({'iqa.'+k:v for k,v in old.state_dict().items()})
        fresh={k:v.clone() for k,v in new.align_region_word.state_dict().items()}
        with patch('torch.load',return_value={'model':source}):load_pretrained_backbone(new,'fake.pth',load_scope='representation')
        for k,v in source.items():torch.testing.assert_close(new.state_dict()[k.removeprefix('iqa.')],v,rtol=0,atol=0)
        for k,v in fresh.items():torch.testing.assert_close(v,new.align_region_word.state_dict()[k],rtol=0,atol=0)
        bad=dict(source);bad.pop('iqa.backbone.fusion.semantic_proj.weight')
        with patch('torch.load',return_value={'model':bad}),self.assertRaises(RuntimeError):load_pretrained_backbone(new,'fake.pth',load_scope='representation')
        x=torch.rand(3,3,16,16);prompts=['a','b','c']
        a=old.forward_single(x,prompts)
        with patch.object(new.align_patch_adapter,'forward',side_effect=AssertionError('unused AP')),patch.object(new.align_patch_img_proj,'forward',side_effect=AssertionError('unused WI')):
            b=new.forward_single(x,prompts)
        for k in ('s_quality','s_quality_global','s_quality_patch','s_align_global'):
            torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        self.assertEqual(b['align_patch_weights'].shape,(3,4))
        params=list(new.backbone.local_encoder.parameters())
        grads=torch.autograd.grad(b['s_align'].sum(),params,allow_unused=True,retain_graph=True)
        self.assertTrue(all(g is None for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in torch.autograd.grad(b['s_quality'].sum(),params,allow_unused=True,retain_graph=True) if g is not None),0)
        targets=[new.align_region_word.image_projection.weight,new.align_region_word.text_projection.weight,new.backbone.text_encoder.context_vectors]
        self.assertTrue(all(g.abs().sum()>0 for g in torch.autograd.grad(b['s_align'].sum(),targets,retain_graph=True)))
        cc['train']=study.make_runner().config('aregionword','agiqa3k')['train'];opt=build_optimizer(cc,new)
        owner={id(p):g for g in opt.param_groups for p in g['params']}
        for p in new.align_region_word.parameters():self.assertEqual(owner[id(p)]['name'],'align');self.assertEqual(owner[id(p)]['lr'],1e-4)
        self.assertEqual(owner[id(new.backbone.fusion.semantic_proj.weight)]['lr'],1e-5)
        self.assertEqual(owner[id(new.shared_text_projector.net[0].weight)]['lr'],1e-4)
        self.assertEqual(len(new_region_parameter_record(new)),2)
        for change in ({'train':{'use_joint_iqa_pretraining':True}}, {'model':{'align_local_temperature':.5}}):
            bad=copy.deepcopy(cc)
            for section,values in change.items():bad[section].update(values)
            with self.assertRaises(ValueError):model(bad)


class GradientTests(unittest.TestCase):
    def test_conflict_one_sided_and_no_grad_rng_buffer_mutation(self):
        m=nn.Module();m.backbone=nn.Module()
        m.backbone.global_encoder=nn.Linear(2,1,bias=False)
        m.backbone.local_encoder=nn.Linear(2,1,bias=False)
        m.backbone.fusion=nn.Module();m.backbone.fusion.semantic_proj=nn.Linear(2,1,bias=False)
        x=torch.ones(1,2);q=m.backbone.global_encoder(x).sum()+m.backbone.local_encoder(x).sum()
        a=-2*m.backbone.global_encoder(x).sum()
        for p in m.parameters():p.grad=torch.ones_like(p)*17
        rng=torch.random.get_rng_state();state=copy.deepcopy(m.state_dict());stats=probe_task_gradients(m,q,a)
        self.assertEqual(stats,probe_task_gradients(m,q,a,loss_scale=1024))
        self.assertAlmostEqual(stats['clip_visual']['cosine'],-1);self.assertAlmostEqual(stats['clip_visual']['align_over_quality'],2)
        self.assertIsNone(stats['resnet_local']['cosine']);self.assertEqual(stats['resnet_local']['quality_only_tensors'],1)
        self.assertEqual(stats['global_projection']['absent_tensors'],1)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
        for k,v in state.items():torch.testing.assert_close(m.state_dict()[k],v,rtol=0,atol=0)
        for p in m.parameters():torch.testing.assert_close(p.grad,torch.ones_like(p)*17,rtol=0,atol=0)
        (q+a).backward() # graph remains live

    def test_probe_same_training_update_and_schedule(self):
        c=config();c['train'].update(pcgrad={'enabled':False})
        m=model(c).train();other=copy.deepcopy(m).train();x=torch.rand(3,3,16,16)
        rng=torch.random.get_rng_state()
        a=m.forward_single(x,['a','b','c']);q=a['s_quality'].sum();align=a['s_align'].sum()
        with tempfile.TemporaryDirectory() as d:
            cc={'output_dir':d,'train':{'task_gradient_probe':{'version':'same_forward_v1','epochs':[1],'batches':1}}}
            maybe_record_probe(m,{'loss_quality_task':q,'loss_align_task':align},cc,1,0)
            maybe_record_probe(m,{},cc,1,1)
            self.assertEqual(len((Path(d)/'task_gradient_probe.jsonl').read_text().splitlines()),1)
        (q+align).backward();torch.random.set_rng_state(rng)
        b=other.forward_single(x,['a','b','c']);(b['s_quality'].sum()+b['s_align'].sum()).backward()
        for (n,p),(n2,p2) in zip(m.named_parameters(),other.named_parameters()):
            self.assertEqual(n,n2)
            if p.grad is not None:torch.testing.assert_close(p.grad,p2.grad,rtol=0,atol=0)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)


    def test_real_trainer_hook_uses_existing_forward_and_preserves_adam_update(self):
        from train_cm_iqa import train_one_epoch, compute_losses
        from test_downstream_pipeline_controls import batch
        for region in (False, True):
            c=config()
            c['train']=copy.deepcopy(study.make_runner().config('aprobe','agiqa3k')['train'])
            if region:c['model']['align_region_word']={'version':'clip_region_word_v1'}
            baseline=model(c);observed=copy.deepcopy(baseline)
            opt=torch.optim.AdamW(baseline.parameters(),lr=1e-4)
            opt2=torch.optim.AdamW(observed.parameters(),lr=1e-4)
            data=batch(['a','b','c','d']);data['image']=torch.rand(4,3,16,16)
            rng=torch.random.get_rng_state()
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                plain=copy.deepcopy(c);plain['output_dir']=d;plain['train'].pop('task_gradient_probe')
                train_one_epoch(baseline,[data],opt,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),plain,1)
                torch.random.set_rng_state(rng);c['output_dir']=d
                with patch('train_cm_iqa.compute_losses',wraps=compute_losses) as calls:
                    train_one_epoch(observed,[data],opt2,torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),c,1)
                self.assertEqual(calls.call_count,1)
                record=json.loads((Path(d)/'task_gradient_probe.jsonl').read_text())
                if region:self.assertIsNone(record['groups']['resnet_local']['cosine'])
            for k,v in baseline.state_dict().items():torch.testing.assert_close(v,observed.state_dict()[k],rtol=0,atol=0)


class DiagnosticAuditTests(unittest.TestCase):
    def rows(self):
        return [dict(version='same_forward_v1',model_mode='train',epoch=e,batch=b,
                     groups={k:dict(status='ok',quality_norm=1.,align_norm=0.,cosine=None)
                             for k in ('clip_visual','resnet_local','global_projection')})
                for e in (1,10,30,60) for b in (1,2)]

    def test_actual_infinite_autograd_is_unavailable_without_fabricating_values(self):
        from utils.task_gradient_audit import audit_probe_records
        m=nn.Module();m.backbone=nn.Module();m.backbone.global_encoder=nn.Linear(1,1,bias=False)
        p=m.backbone.global_encoder.weight;p.grad=torch.ones_like(p)*17
        stats=probe_task_gradients(m,p.sum()*float('inf'),p.sum())
        self.assertEqual(stats['clip_visual']['status'],'nonfinite')
        self.assertIsNone(stats['clip_visual']['quality_norm']);self.assertIsNone(stats['clip_visual']['cosine'])
        self.assertEqual(p.grad.item(),17)
        rows=self.rows();rows[0]['groups']=stats
        report=audit_probe_records(rows,'aprobe')
        self.assertEqual(report['unavailable_group_records'],3)
        self.assertEqual(report['diagnostics_status'],'partial')
        self.assertEqual(rows[0]['groups']['clip_visual']['status'],'nonfinite')

    def test_malformed_status_schedule_finite_claims_and_resnet_dependency_still_fail(self):
        from utils.task_gradient_audit import audit_probe_records
        for case in ('status','schedule','nan_ok','nonfinite_numeric','resnet','nonfinite_resnet'):
            rows=self.rows();g=rows[0]['groups']['clip_visual']
            if case=='status':g['status']='unknown'
            elif case=='schedule':rows.pop()
            elif case=='nan_ok':g['quality_norm']=float('nan')
            elif case=='nonfinite_numeric':g['status']='nonfinite'
            elif case=='resnet':rows[0]['groups']['resnet_local'].update(align_norm=1,cosine=.5)
            else:rows[0]['groups']['resnet_local']=dict(status='nonfinite',quality_norm=None,align_norm=None,cosine=None,shared_tensors=1)
            with self.assertRaises(ValueError):audit_probe_records(rows,'aregionword')
        rows=self.rows()
        for row in rows:
            row['groups']={k:{'status':'no_trainable_parameters'} for k in row['groups']}
        self.assertEqual(audit_probe_records(rows,'aprobe')['diagnostics_status'],'unavailable')


class SelectionTests(unittest.TestCase):
    def test_mean_policy_selects_on_validation_and_restores_separate_epochs(self):
        c={'dataset':{'type':'agiqa3k'},'train':{'compare_selection_policies':True,'selection_comparison_version':'quality_mean_v1'}}
        configure_selection_comparison(c,SimpleNamespace(selection_metric=None))
        tracker=SelectionComparison(c['train']['selection_policies']);m=nn.Linear(1,1)
        for epoch,q,a in ((1,.8,.8),(2,.9,.4),(3,.9,.4)):
            tracker.update(m,dict(val_quality_SROCC=q,val_align_SROCC=a,val_align_N=10),epoch,get_selection_value)
        self.assertEqual(tracker.best['quality']['epoch'],2);self.assertEqual(tracker.best['joint']['epoch'],1)
        results=tracker.evaluate_selected(m,lambda:({}, {'test_align_SROCC':-1}))
        self.assertEqual(results['joint']['best_epoch'],1)
        for bad in (dict(val_quality_SROCC=.8,val_align_N=0),dict(val_quality_SROCC=.8,val_align_SROCC=float('nan'),val_align_N=10)):
            with self.assertRaises(ValueError):get_selection_value(bad,{'train':{'selection_metric':'val_quality_align_mean_SROCC'}})


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.r=study.make_runner();p=patch.object(fixture,'r',self.r);p.start();self.addCleanup(p.stop)
        self.f=fixture.RunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        shutil.copy(study.ROOT/'docs/align_representation.md',self.f.root/'docs/align_representation.md')

    def artifact(self,v,ds):
        r=self.r;d,t=self.f.fake(v,ds,False)
        with t.open() as f:rows=list(csv.DictReader(f))
        rows=[x for x in rows if x['event'] not in ('best_val','best_test','final')]
        for row in rows:
            if row['event']=='epoch':m=json.loads(row['metrics_json']);m['val_val_align_N']=10;row['metrics_json']=json.dumps(m)
        result={}
        for name,key in r.config(v,ds)['train']['selection_policies'].items():
            epoch=40 if name=='quality' else 30 # mean ties epochs30..40 => first30
            result[name]=dict(best_epoch=epoch,selection_metric=key,best_selection_value=1 if name=='quality' else .95,best_test={'test_align_SROCC':.7},best_val={'val_quality_SROCC':1 if name=='quality' else .9,'val_align_SROCC':.9 if name=='quality' else 1})
            rows.append(dict(event=name+'_selected',epoch=str(epoch),metrics_json=json.dumps(result[name])))
        rows.append(dict(event='final',epoch='60',metrics_json=json.dumps(dict(last_epoch=60,comparison_complete=True,selection_policies=result))))
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        records=[dict(version='same_forward_v1',model_mode='train',epoch=e,batch=b,groups={k:{'status':'ok','quality_norm':1.,'align_norm':0.,'cosine':None} for k in ('clip_visual','resnet_local','global_projection')}) for e in (1,10,30,60) for b in (1,2)]
        (d/'task_gradient_probe.jsonl').write_text('\n'.join(json.dumps(x) for x in records)+'\n')
        if v=='aregionword':
            m=r.base.read_json(d/'run_metadata.json');m['region_word_version']='clip_region_word_v1';m['region_word_initialization']=[dict(name='align_region_word.'+n+'.weight',shape=[256,width],initialization='new_random',sha256='a'*64) for n,width in [('image_projection',1024),('text_projection',768)]];(d/'run_metadata.json').write_text(json.dumps(m))
        self.f.seal(d,t);return d,t

    def test_four_artifacts_pack_and_selection_tampering_refusal(self):
        r=self.r;controls={}
        for ds in r.DATASETS:
            for v in r.VARIANTS:
                d,t=self.artifact(v,ds);r.verify(v,ds,None,self.f.source);controls.setdefault(ds,(d,t))
        with patch.object(r,'control',side_effect=lambda ds,split,source:controls[ds]),contextlib.redirect_stdout(io.StringIO()):r.pack()
        import tarfile
        with tarfile.open(self.f.root/f'{r.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.jsonl') for n in names),4);self.assertFalse(any(n.endswith('.pth') for n in names))
        d,t=r.paths('aregionword','agiqa3k');m=r.base.read_json(d/'run_metadata.json');m.pop('region_word_initialization');(d/'run_metadata.json').write_text(json.dumps(m));self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'initialization'):r.verify('aregionword','agiqa3k',None,self.f.source)

    def make_probe_nonfinite(self,d):
        path=d/'task_gradient_probe.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['groups']['clip_visual']={
            'status':'nonfinite','quality_norm':None,'align_norm':None,
            'cosine':None,'align_over_quality':None,
            'shared_tensors':1,'quality_only_tensors':0,'align_only_tensors':0,'absent_tensors':0}
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows))

    def test_nonfinite_diagnostic_preserves_completed_files_and_continues_only_pending_dataset(self):
        r=self.r
        for queue,v in (('probe','aprobe'),('region','aregionword')):
            d,t=self.artifact(v,'agiqa3k');self.make_probe_nonfinite(d);self.f.seal(d,t)
            before={p:p.read_bytes() for p in d.iterdir() if p.is_file()};before[t]=t.read_bytes()
            with patch.object(r,'control'),patch.object(r,'run') as launch,contextlib.redirect_stdout(io.StringIO()) as log:
                r.run_all(queue,continue_completed=True)
            self.assertEqual([call.args for call in launch.call_args_list],[(v,'aigciqa2023')])
            self.assertIn('DIAGNOSTIC WARNING',log.getvalue());self.assertIn('skip',log.getvalue())
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            # A genuine result/file integrity failure still blocks continuation.
            (d/'console.log').write_text('changed')
            with patch.object(r,'control'),patch.object(r,'run') as launch,self.assertRaises(ValueError):
                r.run_all(queue,continue_completed=True)
            launch.assert_not_called()

    def test_nonfinite_probe_is_retained_in_pack_and_explicitly_marked_partial(self):
        r=self.r;controls={}
        for ds in r.DATASETS:
            for v in r.VARIANTS:
                d,t=self.artifact(v,ds);self.make_probe_nonfinite(d);self.f.seal(d,t);controls.setdefault(ds,(d,t))
        with patch.object(r,'control',side_effect=lambda ds,split,source:controls[ds]),contextlib.redirect_stdout(io.StringIO()):r.pack()
        import tarfile
        with tarfile.open(self.f.root/f'{r.VERSION}_results.tar.gz') as tar:
            summary=json.load(tar.extractfile('task_gradient_audit.json'))
            self.assertEqual(len(summary['runs']),4)
            for ds in r.DATASETS:
                for v in r.VARIANTS:
                    c=r.config(v,ds);d,t=r.paths(v,ds);report=summary['runs'][c['run_id']]
                    self.assertTrue(report['training_results_verified'])
                    self.assertEqual(report['diagnostics_status'],'partial')
                    self.assertEqual(report['unavailable_group_records'],1)
                    self.assertEqual(report['valid_group_records'],23)
                    self.assertEqual(report['raw_probe_sha256'],r.base.digest(d/'task_gradient_probe.jsonl'))
                    self.assertEqual(tar.extractfile(c['output_dir']+'/task_gradient_probe.jsonl').read(),(d/'task_gradient_probe.jsonl').read_bytes())

    def test_dry_run_cpu_refusal_no_overwrite_and_exact_config_difference(self):
        r=self.r
        for ds in r.DATASETS:
            a=r.config('aprobe',ds);b=r.config('aregionword',ds)
            for key in ('run_id','output_dir','results_csv'):a[key]=b[key]
            a['model']['align_region_word']={'version':'clip_region_word_v1'};self.assertEqual(a,b)
        with patch.object(r,'source',side_effect=AssertionError('no weights')),contextlib.redirect_stdout(io.StringIO()):r.run_all('all',True)
        with patch.object(r,'control'),patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('aprobe','agiqa3k')
        d,t=self.artifact('aprobe','agiqa3k')
        with patch.object(r,'control'),self.assertRaises(FileExistsError):r.run_all('probe')

if __name__=='__main__':unittest.main()
