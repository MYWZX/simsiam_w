"""Fake process trees and checkpoints only; never signal real processes or train."""
import contextlib,copy,csv,io,json,os,signal,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch,yaml
from scripts import run_pretrain_objectives as study,objective_recovery as handoff,nss_recovery,stop_objective_training as stopper
import test_pretrain_objectives as fixtures
import test_nss_recovery as recovery_fixtures

class ObjectiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.PretrainObjectiveTests('test_completed_source_tamper_and_pack');f.setUp();self.addCleanup(f.doCleanups)
        self.fixture=f;self.root=f.root
        p=patch.object(nss_recovery,'verify_training_code',return_value={'training_code_matches_original':True,'original_code_commit':'fake'});p.start();self.addCleanup(p.stop)

    def interrupted(self):
        r=study.make_runner('pa100')
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        cfg=r.validate('pretrain');d,table=r.paths('pretrain');(d/'completion_manifest.json').unlink()
        selected=d/'best_component_anchor_pretrain.pth';c=torch.load(selected,weights_only=False);c['epoch']=5
        h=cfg['model']['nss_text_gate']['gate_hidden_dim'];e=cfg['model']['fusion']['embed_dim']
        for key,shape in {'nss_proj.0.weight':(h,6),'nss_proj.0.bias':(h,),'nss_proj.2.weight':(e,h),'nss_proj.2.bias':(e,),
                          'nss_gate.0.weight':(h,e+6),'nss_gate.0.bias':(h,),'nss_gate.3.weight':(1,h),'nss_gate.3.bias':(1,)}.items():c['model']['iqa.'+key]=torch.zeros(shape)
        torch.save(c,selected);full=copy.deepcopy(c);full.pop('checkpoint_scope');full.update(epoch=16,optimizer={},scheduler={},scaler={},
            training_state={'best':.05,'best_epoch':5,'no_improve':11},auxiliary_bests={'component_anchor':{'value':.05,'epoch':5}},
            rng={'python':None,'numpy':None,'torch':torch.get_rng_state(),'cuda':None})
        torch.save(full,d/'last_pretrain.pth')
        with table.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json','run_id','best_epoch']);w.writeheader()
            for e in range(1,17):w.writerow(dict(event='epoch',epoch=e,run_id=cfg['run_id'],best_epoch=min(e,5),metrics_json=json.dumps({'val_component_anchor_SROCC':e/100 if e<=5 else .01,'no_improve':max(0,e-5)})))
        return r

    def test_legacy_unsigned_recovery_on_new_visible_gpu_preserves_history(self):
        r=self.interrupted();d,table=r.paths('pretrain');original=table.read_bytes()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',recovery_fixtures.NSSRecoveryTests.fake_resume(self,r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):source=nss_recovery.recover(r,'pa100')
        self.assertEqual(source['training_continuity'],'interruption_recovery_v1');self.assertEqual(source['selected_epoch'],5)
        self.assertEqual((d/'recovery_inputs/results.csv').read_bytes(),original)
        self.assertEqual(r.read_json(d/'recovery_manifest.json')['recovery_cuda_visible_devices'],'0')
        self.assertEqual(r.completed('pretrain'),source)

    def test_wrong_nss_or_missing_weights_refused(self):
        r=self.interrupted();d,_=r.paths('pretrain');p=d/'last_pretrain.pth';original=torch.load(p,weights_only=False)
        for key in ('iqa.nss_descriptor_signature','iqa.nss_pathway_signature'):
            bad=copy.deepcopy(original);bad['model'][key]=torch.tensor([1,6]);torch.save(bad,p)
            with self.assertRaisesRegex(ValueError,'versioned NSS'):nss_recovery.preflight(r,'pa100')
        bad=copy.deepcopy(original);del bad['model']['iqa.nss_gate.0.weight'];torch.save(bad,p)
        with self.assertRaisesRegex(ValueError,'missing/mismatched'):nss_recovery.preflight(r,'pa100')

class WaitAndQueueTests(unittest.TestCase):
    def test_wait_checks_whole_low_queue_and_does_not_use_gpu1(self):
        fake=SimpleNamespace(ROOT=Path('/repo'),QUEUES={'low':('a','b','c','d','e')},STAGES=('pretrain','agiqa1k','agiqa3k','aigciqa2023'),make_runner=Mock())
        with patch.object(handoff,'low_queue_processes',return_value=[(10,'start')]),patch.object(handoff,'process_identity',side_effect=['start',None]),patch.object(handoff.time,'sleep') as sleep,contextlib.redirect_stdout(io.StringIO()):handoff.wait_for_low(fake)
        self.assertEqual(fake.make_runner.call_count,5);self.assertEqual(fake.make_runner.return_value.completed.call_count,20);sleep.assert_called_once_with(15)
        fake.make_runner.return_value.completed.side_effect=ValueError('incomplete')
        with patch.object(handoff,'low_queue_processes',return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'handoff refused'):handoff.wait_for_low(fake)
        fake.make_runner.reset_mock()
        with patch.object(handoff,'low_queue_processes',return_value=[(1,'a'),(2,'b')]):
            with self.assertRaisesRegex(RuntimeError,'Multiple'):handoff.wait_for_low(fake)
        fake.make_runner.assert_not_called()

    def test_recovery_queue_skips_completed_recovers_only_interrupted_and_preflights_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runners={}
            for v in ('done','interrupted','fresh'):
                r=Mock();r.paths.side_effect=lambda stage,v=v:(root/v/stage,root/v/(stage+'.csv'));runners[v]=r
            (root/'done/pretrain').mkdir(parents=True);(root/'done/pretrain/completion_manifest.json').write_text('{}')
            (root/'interrupted/pretrain').mkdir(parents=True)
            fake=SimpleNamespace(ROOT=root,QUEUES={'high':tuple(runners)},STAGES=('pretrain','agiqa1k'),make_runner=lambda v:runners[v],pipeline=SimpleNamespace(run_pipeline=Mock()))
            with patch.object(nss_recovery,'preflight') as preflight,patch.object(nss_recovery,'recover') as resume,contextlib.redirect_stdout(io.StringIO()):handoff.recover_queue(fake,'high')
            preflight.assert_called_once_with(runners['interrupted'],'interrupted');resume.assert_called_once_with(runners['interrupted'],'interrupted')
            self.assertEqual(fake.pipeline.run_pipeline.call_count,3)
            fake.pipeline.run_pipeline.reset_mock()
            with patch.object(nss_recovery,'preflight',side_effect=ValueError('bad checkpoint')),patch.object(nss_recovery,'recover') as resume,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):handoff.recover_queue(fake,'high')
                resume.assert_not_called();fake.pipeline.run_pipeline.assert_not_called()

class TargetedStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'repo';self.proc=Path(self.tmp.name)/'proc'
        c=self.root/'configs/pretrain_objectives/pa100/pretrain.yaml';c.parent.mkdir(parents=True);c.write_text(yaml.safe_dump({'output_dir':'outputs/run','results_csv':'outputs/run.csv'}))
        d=self.root/'outputs/run';d.mkdir(parents=True);(d/'last_pretrain.pth').write_bytes(b'preserve')
        (self.root/'outputs/run.csv').write_text('event,epoch\nepoch,17\n')
        self.create(50,1,'S',['python','scripts/run_pretrain_objectives.py','high','all'])
        self.create(100,50,'T',['python','train_cm_pretrain.py','--config',str(c)])
        self.create(101,100,'T',['python','train_cm_pretrain.py','--config',str(c)])
        self.create(999,1,'R',['/home1/wxl/python','other.py'])

    def create(self,pid,ppid,state,args,gpu='1',uid=None,start='123'):
        p=self.proc/str(pid);p.mkdir(parents=True,exist_ok=True)
        (p/'status').write_text(f'PPid:\t{ppid}\nUid:\t{os.getuid() if uid is None else uid}\nState:\t{state} (state)\n')
        (p/'cmdline').write_bytes(('\0'.join(args)+'\0').encode());(p/'environ').write_bytes(f'CUDA_VISIBLE_DEVICES={gpu}\0'.encode())
        (p/'stat').write_text(f'{pid} (python) {state} '+' '.join(['0']*18)+' '+start)
        if not (p/'cwd').exists():(p/'cwd').symlink_to(self.root,target_is_directory=True)

    def test_only_owned_queue_workers_and_main_are_signalled_in_safe_order(self):
        plan=stopper.inspect(100,self.root,self.proc);send=Mock();stopper.execute(plan,self.proc,send)
        self.assertEqual([c.args for c in send.call_args_list],[(50,signal.SIGTERM),(101,signal.SIGKILL),(100,signal.SIGKILL)])
        self.assertEqual((self.root/'outputs/run/last_pretrain.pth').read_bytes(),b'preserve')
        self.assertNotIn(999,[c.args[0] for c in send.call_args_list])

    def test_wrong_gpu_unpaused_or_reused_pid_is_not_killed(self):
        (self.proc/'100/environ').write_bytes(b'CUDA_VISIBLE_DEVICES=0\0')
        with self.assertRaises(ValueError):stopper.inspect(100,self.root,self.proc)
        (self.proc/'100/environ').write_bytes(b'CUDA_VISIBLE_DEVICES=1\0')
        args=stopper.snapshot(100,self.proc)['argv']
        self.create(100,50,'R',args)
        with self.assertRaises(ValueError):stopper.inspect(100,self.root,self.proc)
        self.create(100,50,'T',args,uid=os.getuid()+1)
        with self.assertRaises(ValueError):stopper.inspect(100,self.root,self.proc)
        self.create(100,50,'T',args)
        plan=stopper.inspect(100,self.root,self.proc);(self.proc/'101/stat').write_text('101 (python) T '+' '.join(['0']*18)+' 999')
        send=Mock()
        with self.assertRaisesRegex(ValueError,'identity changed'):stopper.execute(plan,self.proc,send)
        self.assertNotIn(101,[c.args[0] for c in send.call_args_list]);self.assertNotIn(999,[c.args[0] for c in send.call_args_list])

if __name__=='__main__':unittest.main()
