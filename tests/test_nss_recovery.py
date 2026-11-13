"""Interrupted-run recovery protocol tests with fake artifacts, never real training."""
import contextlib,copy,csv,io,json,os,tarfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from scripts import run_nss_ablation as sweep,nss_recovery as recovery
import test_nss_pipeline as pipeline_tests

REAL_VERIFY_CODE=recovery.verify_training_code

class NSSRecoveryTests(unittest.TestCase):
    def setUp(self):
        f=pipeline_tests.NSSPipelineTests('test_completed_source_tamper_and_pack');f.setUp()
        self.addCleanup(f.doCleanups);self.fixture=f;self.root=f.root
        mocked=patch.object(recovery,'verify_training_code',return_value={'original_code_commit':'fake','training_code_matches_original':True});mocked.start();self.addCleanup(mocked.stop)

    def interrupted(self,variant):
        r=sweep.make_runner(variant)
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('pretrain')
        cfg=r.validate('pretrain');d,table=r.paths('pretrain');(d/'completion_manifest.json').unlink()
        p=d/'best_component_anchor_pretrain.pth';ck=torch.load(p,weights_only=False);ck['epoch']=5;torch.save(ck,p)
        full=copy.deepcopy(ck);full.pop('checkpoint_scope');full.update(epoch=16,optimizer={'state':{},'param_groups':[]},scheduler={'last_epoch':16},
            training_state={'best':.05,'best_epoch':5,'no_improve':11},auxiliary_bests={'component_anchor':{'value':.05,'epoch':5}},
            scaler={},rng={'python':None,'numpy':None,'torch':torch.get_rng_state(),'cuda':None})
        torch.save(full,d/'last_pretrain.pth')
        with table.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json','run_id','best_epoch']);w.writeheader()
            for e in range(1,17):w.writerow(dict(event='epoch',epoch=e,run_id=cfg['run_id'],best_epoch=min(e,5),metrics_json=json.dumps({'val_component_anchor_SROCC':e/100 if e<=5 else .01,'no_improve':max(0,e-5)})))
        return r

    def fake_resume(self,r,fail=False):
        outer=self
        class Process:
            def __init__(self,command,**kwargs):
                outer.assertEqual(command[1],'train_cm_pretrain.py');outer.assertIn('--resume',command)
                d,table=r.paths('pretrain');cfg=r.validate('pretrain');source=Path(command[command.index('--resume')+1])
                state=torch.load(source,weights_only=False)
                outer.assertEqual(state['epoch'],16);outer.assertEqual(state['training_state']['no_improve'],11)
                with table.open('a',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json','run_id','best_epoch'])
                    def write(event,e,metrics):w.writerow(dict(event=event,epoch=e,metrics_json=json.dumps(metrics),run_id=cfg['run_id'],best_epoch=5))
                    write('resume',16,{'checkpoint':str(source),'start_epoch':17})
                    if not fail:
                        for e in range(17,21):write('epoch',e,{'val_component_anchor_SROCC':.01,'no_improve':e-5})
                        write('early_stop',20,{'patience':15})
                        write('final',20,{'selection_metric':'val_component_anchor_SROCC','last_epoch':20,'best_component_anchor_epoch':5,'best_component_anchor_checkpoint':str(d/'best_component_anchor_pretrain.pth')})
                # Atomic publication changes the current checkpoint, not preserved inode.
                temporary=d/'last_pretrain.new';torch.save({'epoch':20},temporary);temporary.replace(d/'last_pretrain.pth')
                self.stdout=iter(['fake resume complete\n'])
            def wait(self):return 1 if fail else 0
        return Process

    def run_recovery(self,r,variant,fail=False):
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fake_resume(r,fail)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):return recovery.recover(r,variant)

    def test_dry_run_cpu_guard_and_active_run_are_read_only(self):
        r=self.interrupted('stats6');d,table=r.paths('pretrain');before=table.read_bytes()
        with patch.object(r.subprocess,'Popen',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):recovery.recover(r,'stats6',True)
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):recovery.recover(r,'stats6')
        self.assertFalse((d/'recovery_inputs').exists());self.assertEqual(table.read_bytes(),before)
        with patch.object(recovery,'active_trainer',return_value=123):
            with self.assertRaisesRegex(RuntimeError,'123'):recovery.preflight(r,'stats6')

    def test_two_variants_preserve_history_selected_state_and_pack_without_weights(self):
        for variant in sweep.VARIANTS:
            r=self.interrupted(variant);d,table=r.paths('pretrain');original=table.read_bytes();oldweights=r.digest(d/'last_pretrain.pth')
            source=self.run_recovery(r,variant)
            self.assertEqual(source['training_continuity'],'interruption_recovery_v1');self.assertEqual(source['selected_epoch'],5)
            self.assertEqual((d/'recovery_inputs/results.csv').read_bytes(),original)
            self.assertEqual(r.digest(d/'recovery_inputs/last_pretrain.pth'),oldweights)
            self.assertEqual(torch.load(d/'last_pretrain.pth',weights_only=False)['epoch'],20)
            self.assertEqual(r.completed('pretrain'),source)
            with contextlib.redirect_stdout(io.StringIO()):r.pack('pretrain')
            with tarfile.open(self.root/f'{r.VERSION}_pretrain_results.tar.gz') as tar:
                names=tar.getnames();self.assertTrue(any(n.endswith('recovery_manifest.json') for n in names));self.assertTrue(any(n.endswith('recovery_inputs/results.csv') for n in names));self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
            with patch.object(r.subprocess,'Popen',side_effect=AssertionError('no repeat')):self.assertEqual(recovery.recover(r,variant),source)
            metadata=r.read_json(d/'recovery_manifest.json');self.assertEqual(metadata['selection_state']['no_improve'],11)
            (d/'recovery_inputs/console.log').write_text('tamper')
            with self.assertRaises(ValueError):r.completed('pretrain')

    def test_failed_recovery_does_not_publish_completion_or_overwrite_second_attempt(self):
        r=self.interrupted('ggd');d,_=r.paths('pretrain')
        with self.assertRaisesRegex(RuntimeError,'recovery failed'):self.run_recovery(r,'ggd',fail=True)
        self.assertFalse((d/'completion_manifest.json').exists());self.assertTrue((d/'recovery_inputs/last_pretrain.pth').exists())
        with self.assertRaises(FileExistsError):recovery.preflight(r,'ggd')
        with self.assertRaises(FileNotFoundError):r.pack('pretrain')

    def test_mismatched_epoch_mode_counters_and_effective_config_refused(self):
        r=self.interrupted('stats6');d,table=r.paths('pretrain');p=d/'last_pretrain.pth';original=torch.load(p,weights_only=False)
        cases=[]
        broken=copy.deepcopy(original);broken['epoch']=15;cases.append(broken)
        broken=copy.deepcopy(original);broken['model']['iqa.nss_descriptor_signature']=torch.tensor([2,2]);cases.append(broken)
        broken=copy.deepcopy(original);broken['training_state']['no_improve']=0;cases.append(broken)
        broken=copy.deepcopy(original);broken['cfg']['train']['lr']=1.;cases.append(broken)
        broken=copy.deepcopy(original);del broken['rng'];cases.append(broken)
        for broken in cases:
            torch.save(broken,p)
            with self.assertRaises(ValueError):recovery.preflight(r,'stats6')
            self.assertFalse((d/'recovery_inputs').exists())

    def test_recovered_source_is_used_by_downstream(self):
        r=self.interrupted('stats6');source=self.run_recovery(r,'stats6')
        class Process:
            def __init__(inner,command,**kwargs):
                cfg=r.validate('agiqa3k');d,table=r.paths('agiqa3k')
                (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                r.write_new(d/'run_metadata.json',{'seed':42,'split_seed':42,'checkpoint_verified_after_load':True,'checkpoint':{'sha256':source['sha256']}})
                with table.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
                    for e in range(1,61):w.writerow({'event':'epoch','epoch':e,'metrics_json':'{}'})
                    w.writerow({'event':'final','epoch':60,'metrics_json':json.dumps({'selection_metric':'val_quality_SROCC','last_epoch':60})})
                inner.stdout=iter(['fake downstream complete\n'])
            def wait(inner):return 0
        with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Process),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):r.run('agiqa3k')
        self.assertEqual(r.completed('agiqa3k'),source)
        d,_=r.paths('agiqa3k');self.assertEqual(r.read_json(d/'source_fingerprint.json'),source)
        self.assertEqual(r.read_json(d/'launch_manifest.json')['source']['training_continuity'],'interruption_recovery_v1')

    def test_cli_recovers_before_downstream_and_failure_stops_chain(self):
        self.addCleanup(os.chdir,Path.cwd())
        r=sweep.make_runner('stats6')
        with patch.object(sweep,'make_runner',return_value=r),patch('sys.argv',['run_nss_ablation.py','stats6','recover-all']),patch.object(recovery,'recover') as resume,patch.object(sweep.pipeline,'run_pipeline') as downstream:
            sweep.main();resume.assert_called_once_with(r,'stats6',False);downstream.assert_called_once_with(r,'continue')
        with patch.object(sweep,'make_runner',return_value=r),patch('sys.argv',['run_nss_ablation.py','stats6','recover-all']),patch.object(recovery,'recover',side_effect=ValueError('bad source')),patch.object(sweep.pipeline,'run_pipeline') as downstream:
            with self.assertRaises(ValueError):sweep.main()
            downstream.assert_not_called()
        with patch.object(sweep,'make_runner',return_value=r),patch('sys.argv',['run_nss_ablation.py','stats6','recover-all','--dry-run']),patch.object(recovery,'recover') as resume,patch.object(sweep.pipeline,'run_pipeline') as downstream:
            sweep.main();resume.assert_called_once_with(r,'stats6',True);downstream.assert_not_called()

    def test_training_code_guard_rejects_changed_or_unknown_source(self):
        r=sweep.make_runner('stats6');launch={'runtime':{'code_commit':'original'}}
        with patch.object(r.subprocess,'run') as run:
            run.return_value.returncode=0;run.return_value.stdout=''
            self.assertTrue(REAL_VERIFY_CODE(r,launch)['training_code_matches_original'])
            run.return_value.stdout='models/nss.py'
            with self.assertRaisesRegex(ValueError,'changed'):REAL_VERIFY_CODE(r,launch)
            run.return_value.returncode=1
            with self.assertRaisesRegex(ValueError,'Cannot verify'):REAL_VERIFY_CODE(r,launch)
        with self.assertRaisesRegex(ValueError,'missing'):REAL_VERIFY_CODE(r,{})

    def test_base_runner_keeps_rejecting_unrecorded_resume(self):
        r=self.interrupted('stats6');d,_=r.paths('pretrain');self.run_recovery(r,'stats6')
        del r.validate_recovery_history
        with self.assertRaisesRegex(ValueError,'non-resumed'):r.completed('pretrain')

if __name__=='__main__':unittest.main()
