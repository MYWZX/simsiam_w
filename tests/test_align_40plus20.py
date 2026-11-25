"""Synthetic same-prefix trajectories, 40/41 boundary, optimizer/LR and strict protocol evidence."""
import contextlib
import copy
import csv
import shutil
import tarfile
import io
import json
import math
import subprocess
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts import run_align_40plus20 as study
from scripts import run_align_representation as peers
from utils import fair_40plus20 as phase
from train_cm_iqa import (train_one_epoch,evaluate,build_optimizer,apply_progressive_finetune_stage,
                         get_selection_value)
from utils.selection_policies import SelectionComparison
from utils.protected_local import frozen_digest
from test_align_representation import model,config as tiny
from test_downstream_pipeline_controls import batch


def config():
    c=tiny();source=study.config('agiqa3k');c['train']=copy.deepcopy(source['train'])
    c['seed']=42;c['split_seed']=42;c['scheduler']=source['scheduler'];c['dataset']=source['dataset']
    c['model']['align_region_word']={'version':'clip_region_word_v1'}
    return c


class PhaseTests(unittest.TestCase):
    def setUp(self):
        state=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,state);torch.manual_seed(42)

    def test_same_first40_then_local20_keep_optimizer_cosine_and_frozen_outputs(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);c=config();c.update(output_dir=str(root/'staged'));Path(c['output_dir']).mkdir()
            # Keep the normal losses/augmentation, and the real first-batch
            # audit; only bound diagnostics to one available synthetic batch.
            c['train']['task_gradient_probe']['batches']=1
            reference=copy.deepcopy(c);reference['train'].pop('fair_40plus20');reference['output_dir']=str(root/'reference');Path(reference['output_dir']).mkdir()
            m=model(c);other=copy.deepcopy(m);opt=build_optimizer(c,m);other_opt=build_optimizer(reference,other)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-6)
            other_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(other_opt,T_max=60,eta_min=1e-6)
            scaler=torch.amp.GradScaler('cpu',enabled=False);other_scaler=torch.amp.GradScaler('cpu',enabled=False)
            initial_trainable={id(p):p.requires_grad for p in m.parameters()}
            other_trainable={id(p):p.requires_grad for p in other.parameters()}
            members=[[id(p) for p in group['params']] for group in opt.param_groups]
            optimizer_identity=id(opt);schedule_identity=id(schedule)
            data=batch(['a','b','c','d']);data['image']=torch.tensor([0.,.2,.6,1.])[:,None,None,None].expand(-1,3,16,16).clone()
            selected=SelectionComparison(c['train']['selection_policies']);boundary=None
            for epoch in range(1,61):
                if epoch<=40:
                    rng=torch.random.get_rng_state()
                    apply_progressive_finetune_stage(other_opt,reference,epoch,other_trainable)
                    train_one_epoch(other,[data],other_opt,other_scaler,torch.device('cpu'),reference,epoch)
                    evaluate(other,[data],torch.device('cpu'),reference,split='val');other_schedule.step()
                    reference_rng=torch.random.get_rng_state();torch.random.set_rng_state(rng)
                apply_progressive_finetune_stage(opt,c,epoch,initial_trainable)
                phase.begin_epoch(m,opt,c,epoch)
                train_one_epoch(m,[data],opt,scaler,torch.device('cpu'),c,epoch)
                va=evaluate(m,[data],torch.device('cpu'),c,split='val');phase.record_validation(m,c,epoch,va);schedule.step()
                if epoch<=40:
                    torch.testing.assert_close(torch.random.get_rng_state(),reference_rng,rtol=0,atol=0)
                    for k,v in other.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
                if epoch==40:boundary=copy.deepcopy(phase.frozen_metrics(va))
                if epoch>=41:
                    self.assertFalse(m.backbone.training);self.assertFalse(m.quality_adapter.training)
                    self.assertFalse(m.backbone.text_encoder.context_vectors.requires_grad)
                    self.assertTrue(m.align_region_word.training is False) # evaluate() ends in eval
                    self.assertTrue(all(n.startswith(phase.PREFIXES) for n,p in m.named_parameters() if p.requires_grad))
                    self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith(phase.PREFIXES)))
                    for k,v in boundary.items():self.assertAlmostEqual(va[k],v,places=6)
                selected.update(m,va,epoch,get_selection_value)
            proof=phase.finalize(m,opt,schedule,c)
            self.assertEqual(proof['frozen_before'],proof['frozen_after'])
            self.assertEqual(set(proof['local_steps_before'].values()),{40})
            self.assertEqual(set(proof['local_steps_after'].values()),{60})
            self.assertEqual(id(opt),optimizer_identity);self.assertEqual(id(schedule),schedule_identity)
            self.assertEqual(members,[[id(p) for p in g['params']] for g in opt.param_groups])
            self.assertTrue(all(abs(lr-1e-6)<1e-12 for lr in proof['lrs_final'].values()))
            # Selection happens after the frozen-state proof; an earlier best
            # state may legally differ from epoch40 and must not invalidate it.
            result=selected.evaluate_selected(m,lambda:({},{}));self.assertEqual(set(result),{'quality','joint'})
            self.assertEqual(json.loads((Path(c['output_dir'])/'fair_phase_manifest.json').read_text()),proof)

    def test_default_noop_and_reject_non60_restarts_or_invalid_boundary(self):
        c=config();m=model(c);opt=build_optimizer(c,m)
        plain=copy.deepcopy(c);plain['train'].pop('fair_40plus20');rng=torch.random.get_rng_state()
        self.assertIsNone(phase.begin_epoch(m,opt,plain,41));self.assertIsNone(phase.finalize(m,None,None,plain))
        torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
        for key,value in [('epochs',80),('compare_selection_policies',False),('pcgrad',{'enabled':True}),('refit_plan','old'),('fair_40plus20',{})]:
            bad=copy.deepcopy(c);bad['train'][key]=value
            with self.assertRaises(ValueError):phase.validate(bad)
        with self.assertRaises(RuntimeError):phase.begin_epoch(m,opt,c,41)
        phase.record_validation(m,c,40,{'val_quality_SROCC':.8,'val_align_global_SROCC':.7})
        with self.assertRaisesRegex(RuntimeError,'accumulated'):phase.begin_epoch(m,opt,c,41)


class ProtocolTests(unittest.TestCase):
    def test_two_configs_only_differ_from_existing_region_by_paths_and_phase_flag(self):
        names=[]
        for ds in ('agiqa3k','aigciqa2023'):
            c=study.config(ds);expected=peers.make_runner().config('aregionword',ds)
            for key in ('run_id','output_dir','results_csv'):expected[key]=c[key]
            expected['train']['fair_40plus20']=dict(phase.OPTION)
            self.assertEqual(c,expected);self.assertEqual(c['train']['epochs'],60);names.append(c['run_id'])
            phase.validate(c)
        self.assertEqual(len(set(names)),2)
        r=study.make_runner()
        with patch.object(r,'source',side_effect=AssertionError('no weights')),contextlib.redirect_stdout(io.StringIO()):r.run_all('all',True)
        with patch.object(r,'source',return_value={}),patch.object(r,'control'),patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):r.run(study.VARIANT,'agiqa3k')

    def evidence(self):
        groups=[{'name':'align','lr':1e-4}];rows=[]
        fixed={'val_quality_SROCC':.8,'val_align_global_SROCC':.7}
        for e in range(1,61):
            m={'lr_align':1e-6+(1e-4-1e-6)*(1+math.cos(math.pi*e/60))/2,
               'train_optimizer_updates':1,**{'val_'+k:v for k,v in fixed.items()}}
            rows.append({'event':'epoch','epoch':str(e),'metrics_json':json.dumps(m)})
        proof=dict(version=phase.VERSION,boundary_epoch=40,start_epoch=41,last_epoch=60,joint_epochs=40,local_epochs=20,
            source_state='current_epoch40_not_validation_selected',optimizer_recreated=False,scheduler_restarted=False,warmup_used=False,
            scheduler_T_max=60,scheduler_last_epoch=60,frozen_before='a'*64,frozen_after='a'*64,boundary_validation=fixed,
            trainable_parameters=[prefix+'weight' for prefix in phase.PREFIXES],
            local_steps_before={prefix+'weight':40 for prefix in phase.PREFIXES},
            local_steps_after={prefix+'weight':60 for prefix in phase.PREFIXES},
            lrs_at_transition={'align':1e-6+(1e-4-1e-6)*.25},lrs_final={'align':1e-6})
        transition={k:v for k,v in proof.items() if k not in ('last_epoch','joint_epochs','local_epochs','frozen_after','local_steps_after','lrs_final','scheduler_T_max','scheduler_last_epoch','trainable_parameters')}
        rows.append({'event':'fair_phase_transition','epoch':'41','metrics_json':json.dumps(transition)})
        return proof,rows,groups

    def test_budget_lr_frozen_state_and_optimizer_reset_evidence_fail_closed(self):
        proof,rows,groups=self.evidence();study.verify_phase(proof,rows,groups)
        for kind in ('budget','lr','optimizer','hash','epoch','metric'):
            p=copy.deepcopy(proof);r=copy.deepcopy(rows)
            if kind=='budget':p['last_epoch']=80
            elif kind=='lr':m=json.loads(r[40]['metrics_json']);m['lr_align']=1e-4;r[40]['metrics_json']=json.dumps(m)
            elif kind=='optimizer':p['local_steps_after'][p['trainable_parameters'][0]]=20
            elif kind=='hash':p['frozen_after']='b'*64
            elif kind=='epoch':p['start_epoch']=40
            else:m=json.loads(r[40]['metrics_json']);m['val_val_quality_SROCC']=.7;r[40]['metrics_json']=json.dumps(m)
            with self.assertRaises(ValueError):study.verify_phase(p,r,groups)

class PackageTests(unittest.TestCase):
    def setUp(self):
        import test_align_representation as fixture_rep
        self.helper=fixture_rep.ProtocolTests();self.helper.setUp();self.addCleanup(self.helper.doCleanups)
        self.peer=self.helper.r;self.root=self.helper.f.root;self.source=self.helper.f.source
        (self.root/'scripts').mkdir();shutil.copy(study.ROOT/'scripts/run_align_tuning.py',self.root/'scripts/run_align_tuning.py')
        shutil.copy(study.ROOT/'docs/align_40plus20.md',self.root/'docs/align_40plus20.md')
        for obj,name,value in ((study,'ROOT',self.root),(study,'peer_runner',lambda:self.peer)):
            patcher=patch.object(obj,name,value);patcher.start();self.addCleanup(patcher.stop)
        self.r=study.make_runner()
        patcher=patch.object(self.r,'source',return_value=self.source);patcher.start();self.addCleanup(patcher.stop)

    def artifact(self,runner,v,ds,staged=False):
        import test_align_tuning as fixture
        with patch.object(fixture,'r',runner):d,t=self.helper.f.fake(v,ds,False)
        meta=runner.base.read_json(d/'run_metadata.json');split=runner.base.read_json(d/'data_split_manifest.json')
        split['train_batches']=1;meta['data_split']={k:x for k,x in split.items() if k!='indices'}
        (d/'data_split_manifest.json').write_text(json.dumps(split))
        if v!='aprobe':
            meta['region_word_version']='clip_region_word_v1';meta['region_word_initialization']=[dict(name='align_region_word.'+n+'.weight',shape=[256,width],initialization='new_random',sha256='a'*64) for n,width in [('image_projection',1024),('text_projection',768)]]
        (d/'run_metadata.json').write_text(json.dumps(meta));groups=meta['optimizer_initialization']
        with t.open() as f:rows=[r for r in csv.DictReader(f) if r['event'] not in ('final','best_val','best_test')]
        for row in rows:
            if row['event']!='epoch':continue
            e=int(row['epoch']);m=json.loads(row['metrics_json'])
            m.update(val_val_align_N=10,train_optimizer_updates=1,train_processed_steps=1,train_amp_skipped_steps=0)
            for g in groups:m['lr_'+g['name']]=1e-6+(g['lr']-1e-6)*(1+math.cos(math.pi*e/60))/2
            m['val_val_align_global_SROCC']=.7
            if staged and e>40:m['val_val_quality_SROCC']=1.
            row['metrics_json']=json.dumps(m)
        results={}
        for policy,metric in runner.config(v,ds)['train']['selection_policies'].items():
            epoch=40 if policy=='quality' else 30
            results[policy]=dict(best_epoch=epoch,selection_metric=metric,best_selection_value=1 if policy=='quality' else .95,
                best_val=dict(val_quality_SROCC=1 if policy=='quality' else .9,val_align_SROCC=.9 if policy=='quality' else 1),best_test={'test_align_SROCC':.8})
            rows.append(dict(event=policy+'_selected',epoch=str(epoch),metrics_json=json.dumps(results[policy])))
        rows.append(dict(event='final',epoch='60',metrics_json=json.dumps(dict(last_epoch=60,comparison_complete=True,selection_policies=results))))
        probe=[dict(version='same_forward_v1',model_mode='train',epoch=e,batch=b,groups={n:dict(status='ok',quality_norm=1,align_norm=0,cosine=None) for n in ('clip_visual','resnet_local','global_projection')}) for e in (1,10,30,60) for b in (1,2)]
        if staged:
            proof,_,_=ProtocolTests().evidence()
            proof['boundary_validation']['val_quality_SROCC']=1.
            proof['lrs_at_transition']={g['name']:1e-6+(g['lr']-1e-6)*.25 for g in groups}
            proof['lrs_final']={g['name']:1e-6 for g in groups}
            (d/'fair_phase_manifest.json').write_text(json.dumps(proof))
            transition={k:x for k,x in proof.items() if k not in ('last_epoch','joint_epochs','local_epochs','frozen_after','local_steps_after','lrs_final','scheduler_T_max','scheduler_last_epoch','trainable_parameters')}
            rows.append(dict(event='fair_phase_transition',epoch='41',metrics_json=json.dumps(transition)))
            for record in probe:
                if record['epoch']==60:record['groups']={k:{'status':'no_trainable_parameters'} for k in record['groups']}
        (d/'task_gradient_probe.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in probe))
        with t.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with patch.object(fixture,'r',runner):self.helper.f.seal(d,t)
        return d,t

    def test_full_records_pack_eight_csv_and_refuse_changed_phase_even_if_resealed(self):
        controls={}
        for ds in self.r.DATASETS:
            for v in self.peer.VARIANTS:
                d,t=self.artifact(self.peer,v,ds);self.peer.verify(v,ds,None,self.source)
            d,t=self.artifact(self.r,study.VARIANT,ds,True);self.r.verify(study.VARIANT,ds,None,self.source)
            history=self.root/'outputs'/('history_'+ds);history.mkdir();history_csv=history.with_suffix('.csv');history_csv.write_text('historical reference')
            controls[ds]=(history,history_csv)
        with patch.object(self.r,'control',side_effect=lambda ds,split,fp:controls[ds]),contextlib.redirect_stdout(io.StringIO()):self.r.pack()
        with tarfile.open(self.root/(study.VERSION+'_results.tar.gz')) as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),8)
            self.assertEqual(sum(n.endswith('fair_phase_manifest.json') for n in names),2)
            self.assertFalse(any(n.endswith('.pth') for n in names));self.assertIn('comparison_protocol.json',names)
        d,t=self.r.paths(study.VARIANT,'agiqa3k');proof=json.loads((d/'fair_phase_manifest.json').read_text());proof['optimizer_recreated']=True;(d/'fair_phase_manifest.json').write_text(json.dumps(proof))
        import test_align_tuning as fixture
        with patch.object(fixture,'r',self.r):self.helper.f.seal(d,t)
        with self.assertRaises(ValueError):self.r.verify(study.VARIANT,'agiqa3k',None,self.source)

class LauncherTests(unittest.TestCase):
    def test_detached_launcher_only_submits_two_new_runs_and_stops_on_bad_inputs(self):
        from test_protected_local_recovery import BackgroundLauncherTests
        for status in ('0','1'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);env=BackgroundLauncherTests().prepare(root)
                env['FAKE_RECOVERY_STATUS']=status
                source=Path(__file__).resolve().parents[1]/'scripts/launch_align_40plus20_background.sh'
                shutil.copy(source,root/'scripts/launch_align_40plus20_background.sh')
                result=subprocess.run(['bash','scripts/launch_align_40plus20_background.sh'],cwd=root,env=env,capture_output=True,text=True)
                if status=='1':
                    self.assertNotEqual(result.returncode,0);self.assertFalse((root/'run_logs').exists());continue
                self.assertEqual(result.returncode,0,result.stderr)
                logs=next((root/'run_logs').glob('align4020_*'))
                for _ in range(100):
                    if all((logs/n).is_file() and (logs/n).stat().st_size for n in ('agiqa3k.log','aigciqa2023.log')):break
                    time.sleep(.01)
                a=(logs/'agiqa3k.log').read_text();b=(logs/'aigciqa2023.log').read_text()
                self.assertIn('gpu:0',a);self.assertIn('agiqa3k',a);self.assertIn('a40local20',a)
                self.assertIn('gpu:1',b);self.assertIn('aigciqa2023',b);self.assertIn('setsid',b)
                self.assertNotIn('aprobe',a+b);self.assertNotIn('warmup',a+b)

if __name__=='__main__':unittest.main()
