"""Fixed-temperature forward math, matched initialization and four-run protocol."""
import contextlib
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import torch
from models.region_word_alignment import RegionWordAlignment
from models.alignment_evidence import spatial_diagnostics
from train_cm_iqa import build_optimizer
from scripts import run_align_fixed_temperature as study
import test_align_tuning as fixture
from test_alignment_evidence import model
from test_align_representation import config as tiny

ROOT=Path(__file__).resolve().parents[1]


class MathTests(unittest.TestCase):
    def test_sharpening_matches_analytic_softmax_and_masks_padding(self):
        matcher=RegionWordAlignment(2,2,2)
        with torch.no_grad():
            matcher.image_projection.weight.copy_(torch.eye(2));matcher.text_projection.weight.copy_(torch.eye(2))
        image=torch.eye(2)[None];text=torch.tensor([[[1.,0.],[0.,1.]]]);mask=torch.tensor([[True,False]])
        results=[]
        for tau in (1.,.1):
            e,t,w,_,_=matcher(image,text,mask,tau,{'mode':'mean'})
            expected=torch.tensor([1.,0.]).div(tau).softmax(0)
            torch.testing.assert_close(w[0,:,0],expected);self.assertEqual(w[0,:,1].sum(),0)
            torch.testing.assert_close(e[0],expected)
            results.append(spatial_diagnostics(w))
        self.assertLess(results[1]['spatial_entropy'],results[0]['spatial_entropy'])
        self.assertGreater(results[1]['spatial_max'],results[0]['spatial_max'])

    def test_actual_models_have_identical_weights_rng_groups_and_quality_only_matching_changes(self):
        r=study.make_runner();models={};rng=[]
        for v in study.VARIANTS:
            c=tiny();full=r.config(v,'agiqa3k');c['train']=full['train']
            c['model']['align_region_word']=full['model']['align_region_word']
            c['model']['align_evidence']=full['model']['align_evidence']
            torch.manual_seed(42);models[v]=model(c).eval();rng.append(torch.random.get_rng_state())
            models[v].backbone.text_encoder.tokenizer.decode=lambda ids:'content'
        a,b=models.values();self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for key,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[key],atol=0,rtol=0)
        torch.testing.assert_close(*rng,atol=0,rtol=0)
        for m in models.values():
            self.assertIsNone(m.align_evidence.temperature_logit)
            self.assertEqual(list(m.align_evidence.parameters()),[])
        image=torch.rand(3,3,16,16);prompts=['a','b','c'];out=[m.forward_single(image,prompts) for m in models.values()]
        for key in ('s_quality','s_quality_global','s_quality_patch','s_align_global'):
            torch.testing.assert_close(out[0][key],out[1][key],atol=0,rtol=0)
        self.assertFalse(torch.equal(out[0]['align_patch_weights'],out[1]['align_patch_weights']))
        self.assertLess(out[1]['alignment_diagnostics']['spatial_entropy'].mean(),out[0]['alignment_diagnostics']['spatial_entropy'].mean())
        groups=[]
        for v,m in models.items():
            opt=build_optimizer(r.config(v,'agiqa3k'),m);owner={id(p):g for g in opt.param_groups for p in g['params']}
            self.assertEqual(owner[id(m.backbone.fusion.semantic_proj.weight)]['lr'],1e-5)
            for p in m.align_region_word.parameters():self.assertEqual(owner[id(p)]['lr'],1e-4)
            out[list(models).index(v)]['s_align'].sum().backward()
            self.assertGreater(m.align_region_word.image_projection.weight.grad.abs().sum(),0)
            self.assertGreater(m.backbone.text_encoder.context_vectors.grad.abs().sum(),0)
            groups.append([(g['name'],g['lr'],sum(p.numel() for p in g['params'])) for g in opt.param_groups])
        self.assertEqual(*groups)

    def test_configs_differ_only_temperature_and_output_identity(self):
        r=study.make_runner();paths=[]
        for ds in r.DATASETS:
            a=r.config('tau100',ds);b=r.config('tau010',ds)
            for k in ('run_id','output_dir','results_csv'):paths.extend([a[k]] if k=='output_dir' else []);b[k]=a[k]
            b['model']['align_evidence']['temperature']['initial']=1.
            self.assertEqual(a,b)
            self.assertNotIn('native_reference',a['model']['align_evidence'])
            self.assertEqual(a['model']['align_evidence']['coverage']['mode'],'mean')
            self.assertEqual(a['model']['nss_text_gate'].get('descriptor','legacy'),'legacy')
            self.assertEqual(a['train']['epochs'],60);self.assertEqual(a['seed'],42);self.assertEqual(a['split_seed'],42)
        for args in [('tau100','agiqa1k'),('bad','agiqa3k'),('tau010','agiqa3k',3407)]:
            with self.assertRaises(ValueError):r.config(*args)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.r=study.make_runner();p=patch.object(fixture,'r',self.r);p.start();self.addCleanup(p.stop)
        self.f=fixture.RunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        shutil.copy(ROOT/'docs/align_fixed_temperature.md',self.f.root/'docs')

    def artifact(self,v,ds):
        r=self.r;d,t=self.f.fake(v,ds,False);c=r.config(v,ds);tau=r.VARIANTS[v]
        split=r.base.read_json(d/'data_split_manifest.json');split.update(train_batches=1,test_ids_sha256='test')
        (d/'data_split_manifest.json').write_text(json.dumps(split))
        meta=r.base.read_json(d/'run_metadata.json');meta['data_split']={k:x for k,x in split.items() if k!='indices'}
        meta['alignment_evidence_initialization']={'version':'align_evidence_v1','config':c['model']['align_evidence'],'parameters':[]}
        meta['region_word_version']='clip_region_word_v1'
        meta['region_word_initialization']=[{'name':'align_region_word.'+name+'.weight','shape':shape,'initialization':'new_random','sha256':sha*64}
            for name,shape,sha in [('image_projection',[256,1024],'a'),('text_projection',[256,768],'b')]]
        (d/'run_metadata.json').write_text(json.dumps(meta));groups=meta['optimizer_initialization']
        rows=[x for x in study.read_rows(t) if x['event'] not in ('best_val','best_test','final')]
        def diagnostic(prefix):
            return {prefix+'evidence_'+k:value for k,value in dict(temperature=tau,spatial_entropy=.8,spatial_max=.1,prompt_truncated=0.,
                    s_align_MAE=.1,s_align_global_MAE=.1,s_align_patch_MAE=.2,token_match_mean=.2,token_match_min=-.1).items()}
        for row in rows:
            if row['event']!='epoch':continue
            e=int(row['epoch']);m=json.loads(row['metrics_json']);m.update(diagnostic('val_val_'))
            m.update(val_val_quality_loss=.02,val_val_align_N=10,train_processed_steps=1,train_optimizer_updates=1,train_amp_skipped_steps=0)
            for g in groups:m['lr_'+g['name']]=1e-6+(g['lr']-1e-6)*(1+math.cos(math.pi*e/60))/2
            row['metrics_json']=json.dumps(m)
        result={}
        for name,key in c['train']['selection_policies'].items():
            e=40 if name=='quality' else 30
            result[name]=dict(best_epoch=e,selection_metric=key,best_selection_value=1 if name=='quality' else .95,
                best_val=dict(diagnostic('val_'),val_quality_SROCC=1 if name=='quality' else .9,val_align_SROCC=.9 if name=='quality' else 1),
                best_test=dict(diagnostic('test_'),test_quality_SROCC=.8,test_align_SROCC=.7))
            rows.append(dict(event=name+'_selected',epoch=e,metrics_json=json.dumps(result[name])))
        rows.append(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,comparison_complete=True,selection_policies=result))))
        self.write_rows(t,rows);self.f.seal(d,t);return d,t

    def write_rows(self,t,rows):
        with t.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

    def test_all_four_verified_and_pack_contains_no_weights(self):
        controls={}
        for ds in self.r.DATASETS:
            for v in study.VARIANTS:
                d,t=self.artifact(v,ds);self.r.verify(v,ds,None,self.f.source)
            hd=self.f.root/'history'/ds;hd.mkdir(parents=True);ht=hd.with_suffix('.csv');ht.write_text('history');controls[ds]=(hd,ht)
        with patch.object(self.r,'control',side_effect=lambda ds,*args:controls[ds]),contextlib.redirect_stdout(io.StringIO()):self.r.pack()
        with tarfile.open(self.f.root/(study.VERSION+'_results.tar.gz')) as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),6)
            self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names));self.assertIn('fixed_temperature_protocol.json',names)
        with patch.object(self.r,'control',side_effect=lambda ds,*args:controls[ds]),self.assertRaises(FileExistsError):self.r.pack()

    def test_wrong_temperature_even_resealed_is_rejected(self):
        d,t=self.artifact('tau010','agiqa3k');rows=study.read_rows(t)
        row=next(x for x in rows if x['event']=='epoch');m=json.loads(row['metrics_json']);m['val_val_evidence_temperature']=1.;row['metrics_json']=json.dumps(m)
        self.write_rows(t,rows);self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'temperature'):self.r.verify('tau010','agiqa3k',None,self.f.source)

    def test_peer_initialization_or_optimizer_count_mismatch_rejected(self):
        a,_=self.artifact('tau100','agiqa3k');b,t=self.artifact('tau010','agiqa3k')
        meta=self.r.base.read_json(b/'run_metadata.json');meta['region_word_initialization'][0]['sha256']='c'*64
        (b/'run_metadata.json').write_text(json.dumps(meta));self.f.seal(b,t)
        with self.assertRaisesRegex(ValueError,'fingerprints'):self.r.verify('tau100','agiqa3k',None,self.f.source)
        meta['region_word_initialization'][0]['sha256']='a'*64;meta['optimizer_initialization'][0]['parameter_count']+=1
        (b/'run_metadata.json').write_text(json.dumps(meta));self.f.seal(b,t)
        # CSV group event must also agree, otherwise the original verifier rejects first.
        rows=study.read_rows(t);rows[0]['metrics_json']=json.dumps({'groups':meta['optimizer_initialization']});self.write_rows(t,rows);self.f.seal(b,t)
        with self.assertRaisesRegex(ValueError,'parameter counts'):self.r.verify('tau100','agiqa3k',None,self.f.source)

    def test_partial_continue_and_cpu_refusal_leave_files_unchanged(self):
        d,t=self.artifact('tau100','agiqa3k');(d/'completion_manifest.json').unlink();before=t.read_bytes()
        with patch.object(self.r,'control'),patch.object(self.r,'run') as launch,self.assertRaisesRegex(RuntimeError,'Incomplete'):
            self.r.run_all('all',continue_completed=True)
        launch.assert_not_called();self.assertEqual(t.read_bytes(),before)
        with patch.object(self.r,'control'),patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):
            self.r.run('tau010','aigciqa2023')
        self.assertFalse(self.r.paths('tau010','aigciqa2023')[0].exists())

    def test_dry_run_never_checks_weights_or_launches(self):
        with patch.object(self.r,'source',side_effect=AssertionError('no weights')),patch.object(self.r,'run',side_effect=AssertionError('no training')),contextlib.redirect_stdout(io.StringIO()):self.r.run_all('all',dry_run=True)

    def test_plot_render_png_pdf_from_logged_quality_and_entropy(self):
        from scripts import plot_align_fixed_temperature as plotting
        for v in study.VARIANTS:self.artifact(v,'agiqa3k')
        with patch.object(plotting,'make_runner',return_value=self.r):plotting.render('agiqa3k',self.f.root/'figure')
        self.assertGreater((self.f.root/'figure.png').stat().st_size,10000)
        self.assertTrue((self.f.root/'figure.pdf').read_bytes().startswith(b'%PDF'))


class LauncherTests(unittest.TestCase):
    def prepare(self,root):
        (root/'scripts').mkdir();shutil.copy(ROOT/'scripts/launch_align_fixed_temperature_background.sh',root/'scripts')
        b=root/'bin';b.mkdir();py=b/'python'
        py.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TRACE"\nif [[ "$*" == *preflight_align* && "${FAIL_PREFLIGHT:-0}" == 1 ]]; then exit 1; fi\n');py.chmod(0o755)
        no=b/'nohup';no.write_text('#!/bin/bash\nprintf "gpu=%s args=%s\\n" "$CUDA_VISIBLE_DEVICES" "$*"\n');no.chmod(0o755)
        return dict(os.environ,ABLATION_PYTHON=str(py),TRACE=str(root/'trace'),PATH=str(b)+':'+os.environ['PATH'])

    def test_active_job_guard_matches_only_this_study_and_excludes_parent_launcher(self):
        from scripts.preflight_align_fixed_temperature import active_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=[(1,['bash','scripts/launch_align_fixed_temperature_background.sh']),
                   (2,['python','scripts/run_align_fixed_temperature.py','unit']),
                   (3,['python','train_cm_iqa.py','--config','/home/outputs/align_fixed_temperature_v1/launch.yaml']),
                   (4,['python','train_cm_iqa.py','--config','/home/outputs/another/launch.yaml'])]
            for pid,args in items:
                d=root/str(pid);d.mkdir();(d/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args))
            self.assertEqual(set(active_jobs(root)),{'2','3'})

    def test_preflight_failure_never_starts_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=self.prepare(root);env['FAIL_PREFLIGHT']='1'
            result=subprocess.run(['bash','scripts/launch_align_fixed_temperature_background.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'run_logs').exists())

    def test_background_two_queues_have_unique_logs_and_pids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=self.prepare(root)
            result=subprocess.run(['bash','scripts/launch_align_fixed_temperature_background.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            folder=next((root/'run_logs').glob('align_fixed_temperature_*'))
            self.assertEqual(len(list(folder.glob('*.pid'))),2)
            self.assertIn('unit.log',result.stdout);self.assertIn('sharp.log',result.stdout)

if __name__=='__main__':unittest.main()
