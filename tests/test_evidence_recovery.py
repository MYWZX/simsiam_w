"""Evidence reboot recovery: full pretraining resume, conservative downstream restart."""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from scripts import run_nss_kurtosis as kurt, nss_recovery as resume
from scripts import recover_align_evidence as down
import test_nss_recovery as fixtures

ROOT=Path(__file__).resolve().parents[1]


class KurtosisResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.NSSRecoveryTests('test_dry_run_cpu_guard_and_active_run_are_read_only')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        shutil.copytree(ROOT/kurt.CONFIG,self.root/kurt.CONFIG)
        p=patch.object(kurt,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def interrupted(self):
        r=kurt.make_runner()
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fixture.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        d,t=r.paths('pretrain');(d/'completion_manifest.json').unlink()
        c=torch.load(d/'best_component_anchor_pretrain.pth',weights_only=False);c['epoch']=8;torch.save(c,d/'best_component_anchor_pretrain.pth')
        full=copy.deepcopy(c)
        full.update(epoch=15,optimizer={},scheduler={},scaler={},rng={'python':None,'numpy':None,'torch':torch.get_rng_state(),'cuda':None},
                    training_state={'best':.08,'best_epoch':8,'no_improve':7},auxiliary_bests={'component_anchor':{'epoch':8,'value':.08}})
        torch.save(full,d/'last_pretrain.pth')
        with t.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json','run_id']);w.writeheader()
            for e in range(1,16):w.writerow(dict(event='epoch',epoch=e,run_id=r.validate('pretrain')['run_id'],metrics_json=json.dumps({'val_component_anchor_SROCC':e/100 if e<=8 else .01,'no_improve':max(0,e-8)})))
        return r

    def test_kurtosis_preflight_matches_real_saved_epoch_and_signature(self):
        r=self.interrupted();d,t=r.paths('pretrain');before=t.read_bytes()
        audit=resume.preflight(r,'kurt6')
        self.assertEqual((audit['saved_epoch'],audit['next_epoch']),(15,16))
        self.assertEqual(audit['selection_state']['best_epoch'],8)
        self.assertFalse((d/'recovery_inputs').exists());self.assertEqual(before,t.read_bytes())
        with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):resume.recover(r,'kurt6')
        c=torch.load(d/'last_pretrain.pth',weights_only=False);c['model']['iqa.nss_descriptor_signature']=torch.tensor([1,6]);torch.save(c,d/'last_pretrain.pth')
        with self.assertRaisesRegex(ValueError,'signature'):resume.preflight(r,'kurt6')

    def test_resume_keeps_original_history_optimizer_source_and_completion_provenance(self):
        r=self.interrupted();d,t=r.paths('pretrain');before=t.read_bytes();oldsha=r.digest(d/'last_pretrain.pth')
        outer=self
        class Process:
            def __init__(self,command,**kwargs):
                source=Path(command[command.index('--resume')+1]);state=torch.load(source,weights_only=False)
                outer.assertEqual(state['epoch'],15);outer.assertEqual(state['training_state']['no_improve'],7)
                with t.open('a',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json','run_id'])
                    def write(event,e,metrics):w.writerow(dict(event=event,epoch=e,run_id=r.validate('pretrain')['run_id'],metrics_json=json.dumps(metrics)))
                    write('resume',15,{'checkpoint':str(source),'start_epoch':16})
                    for e in range(16,24):write('epoch',e,{'val_component_anchor_SROCC':.01,'no_improve':e-8})
                    write('early_stop',23,{'patience':15})
                    write('final',23,{'last_epoch':23,'selection_metric':'val_component_anchor_SROCC','best_component_anchor_epoch':8,
                                      'best_component_anchor_checkpoint':str(d/'best_component_anchor_pretrain.pth')})
                temp=d/'last.new';torch.save({'epoch':23},temp);temp.replace(d/'last_pretrain.pth')
                self.stdout=iter(['synthetic resume\n'])
            def wait(self):return 0
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Process),patch('utils.run_metadata.collect_run_metadata',return_value={}),contextlib.redirect_stdout(io.StringIO()):fp=resume.recover(r,'kurt6')
        self.assertEqual(fp['selected_epoch'],8);self.assertEqual(fp['training_continuity'],'interruption_recovery_v1')
        self.assertTrue(t.read_bytes().startswith(before));self.assertEqual(r.digest(d/'recovery_inputs/last_pretrain.pth'),oldsha)
        self.assertEqual(r.completed('pretrain'),fp)
        (d/'recovery_inputs/console.log').write_text('tampered')
        with self.assertRaises(ValueError):r.completed('pretrain')

    def test_cli_dry_run_does_not_continue_partial_or_launch(self):
        r=self.interrupted();self.addCleanup(os.chdir,Path.cwd())
        with patch.object(kurt,'make_runner',return_value=r),patch('sys.argv',['run_nss_kurtosis.py','recover-all','--dry-run']),patch.object(resume,'recover') as recover,patch.object(kurt.pipeline,'run_pipeline') as pipeline:
            kurt.main();recover.assert_called_once_with(r,'kurt6',True);pipeline.assert_not_called()


class DownstreamRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.r=down.make_recovery('legacy');self.r.protocol.ROOT=self.root
        self.r.stages=lambda:[('diag','agiqa3k'),('diag','aigciqa2023')]
        self.r.stage_paths=lambda v,ds:({'run_id':ds},self.root/ds,self.root/(ds+'.csv'))
        self.r.protocol.source=lambda:{'sha256':'fixed'}
        self.r.protocol.verify=lambda *args:None
        self.r.active_jobs=lambda:[]

    def partial(self,ds):
        c,d,t=self.r.stage_paths('diag',ds);d.mkdir();(d/'launch_config.yaml').write_text(yaml.safe_dump(c))
        t.write_text('event,epoch,metrics_json\nepoch,14,{}\n');return d,t

    def test_completed_retained_partial_renamed_with_journal_and_no_completion_fabricated(self):
        completed,_=self.partial('agiqa3k');(completed/'completion_manifest.json').write_text('{}')
        d,t=self.partial('aigciqa2023');before=t.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):entries=self.r.preserve_interrupted()
        self.assertTrue(completed.is_dir());self.assertFalse(d.exists());self.assertFalse(t.exists())
        saved=next(self.root.glob('aigciqa2023.csv.interrupted_*'));self.assertEqual(saved.read_bytes(),before)
        journal=next((self.root/'run_logs').glob('align_evidence_legacy_recovery_*.json'))
        self.assertTrue(json.loads(journal.read_text())['complete'])
        self.assertEqual([x['status'] for x in entries],['completed','partial'])

    def test_process_guard_allows_parent_launcher_but_rejects_queues_and_trainers(self):
        guard=down.make_recovery('legacy').active_jobs
        root=self.root/'proc';root.mkdir()
        for pid,args in [(1,['bash','scripts/launch_evidence_background.sh','--recover-incomplete']),
                         (2,['bash','scripts/run_evidence_chain.sh','legacy']),
                         (3,['python','train_cm_pretrain.py','--config','source.yaml'])]:
            d=root/str(pid);d.mkdir();(d/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args))
        self.assertEqual({p['pid'] for p in guard(root)},{2,3})

    def test_active_unknown_configuration_and_final_without_seal_all_refused(self):
        d,t=self.partial('agiqa3k')
        self.r.active_jobs=lambda:[{'pid':3}]
        with self.assertRaises(RuntimeError):self.r.preserve_interrupted()
        self.r.active_jobs=lambda:[]
        (d/'launch_config.yaml').write_text('wrong: config')
        with self.assertRaises(ValueError):self.r.preserve_interrupted()
        (d/'launch_config.yaml').write_text(yaml.safe_dump({'run_id':'agiqa3k'}))
        t.write_text('event,epoch,metrics_json\nfinal,60,{}\n')
        with self.assertRaisesRegex(RuntimeError,'Possibly completed'):self.r.preserve_interrupted()
        self.assertTrue(d.exists());self.assertFalse(list(self.root.glob('*.interrupted_*')))


class LauncherTests(unittest.TestCase):
    def prepare(self,root):
        (root/'scripts').mkdir();shutil.copy(ROOT/'scripts/launch_evidence_background.sh',root/'scripts')
        b=root/'bin';b.mkdir()
        py=b/'python';py.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TRACE"\nif [[ "$*" == *recover-pretrain* && "${FAIL_PREFLIGHT:-0}" == 1 ]]; then exit 1; fi\n');py.chmod(0o755)
        no=b/'nohup';no.write_text('#!/bin/bash\nprintf "gpu=%s args=%s\\n" "$CUDA_VISIBLE_DEVICES" "$*"\n');no.chmod(0o755)
        return dict(os.environ,ABLATION_PYTHON=str(py),TRACE=str(root/'trace'),PATH=str(b)+':'+os.environ['PATH'])

    def test_failed_resume_preflight_never_moves_outputs_or_starts_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=self.prepare(root);env['FAIL_PREFLIGHT']='1'
            result=subprocess.run(['bash','scripts/launch_evidence_background.sh','--recover-incomplete'],cwd=root,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'run_logs').exists())
            self.assertNotIn('--execute',(root/'trace').read_text())

    def test_launcher_preflights_then_preserves_and_detaches_recovery_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=self.prepare(root)
            result=subprocess.run(['bash','scripts/launch_evidence_background.sh','--recover-incomplete'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            trace=(root/'trace').read_text();self.assertLess(trace.index('recover-pretrain'),trace.index('--execute'))
            self.assertIn('legacy --execute',trace)
            self.assertEqual(len(list((root/'run_logs').glob('evidence_*/*.pid'))),2)

if __name__=='__main__':unittest.main()
