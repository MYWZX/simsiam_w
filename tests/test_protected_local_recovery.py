"""Phase restart preserves incomplete files and never fabricates completion."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from scripts import recover_protected_local as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.items=[('foundation','agiqa3k'),('warmup','richhf'),('old','agiqa3k')]
        def paths(v,ds):
            c=dict(run_id=v+'_'+ds,train={'epochs':60 if v=='foundation' else 20})
            return c,self.root/c['run_id'],self.root/(c['run_id']+'.csv')
        self.paths=paths
        for obj,name,value in ((recovery,'stage_paths',paths),(recovery,'stages',lambda:self.items),
                               (recovery,'active_jobs',lambda:[]),(recovery.protocol,'ROOT',self.root),
                               (recovery.protocol,'source',lambda:{'sha256':'original'})):
            p=patch.object(obj,name,value);p.start();self.addCleanup(p.stop)

    def partial(self,v,ds):
        c,d,t=self.paths(v,ds);d.mkdir()
        (d/'launch_config.yaml').write_text(yaml.safe_dump(c));(d/'console.log').write_bytes(b'original console')
        t.write_text('event,epoch,metrics_json\nepoch,7,{}\n')
        return d,t

    def test_partial_is_preserved_byte_for_byte_completed_and_sources_untouched(self):
        d,t=self.partial('foundation','agiqa3k');warm,csv=self.partial('warmup','richhf')
        (warm/'completion_manifest.json').write_text('{}')
        before={p:p.read_bytes() for p in warm.iterdir()};before[csv]=csv.read_bytes()
        with patch.object(recovery.protocol,'verify') as verify,contextlib.redirect_stdout(io.StringIO()):entries=recovery.preserve_interrupted()
        verify.assert_called_once_with('warmup','richhf')
        self.assertEqual([e['status'] for e in entries],['partial','completed','pending'])
        self.assertFalse(d.exists());self.assertFalse(t.exists())
        archived=next(self.root.glob(d.name+'.interrupted_*'));archived_csv=next(self.root.glob(t.name+'.interrupted_*'))
        self.assertEqual((archived/'console.log').read_bytes(),b'original console')
        self.assertEqual(archived_csv.read_text(),'event,epoch,metrics_json\nepoch,7,{}\n')
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        journal=json.loads(next((self.root/'run_logs').glob('*.json')).read_text())
        self.assertTrue(journal['complete']);self.assertTrue(all(m['moved'] for m in journal['moves']))
        # Preserved directories are not rescanned as fresh runs on retry.
        with patch.object(recovery.protocol,'verify'),contextlib.redirect_stdout(io.StringIO()):recovery.preserve_interrupted()
        self.assertEqual(len(list(self.root.glob(d.name+'.interrupted_*'))),1)

    def test_final_or_checkpoint_without_seal_blocks_all_moves(self):
        for marker in ('csv_final','best_local.pth','local_checkpoint_manifest.json'):
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                def paths(v,ds):return {'run_id':v},root/v,root/(v+'.csv')
                with patch.object(recovery,'stage_paths',side_effect=paths):
                    for v,ds in self.items[:2]:
                        c,d,t=paths(v,ds);d.mkdir();(d/'launch_config.yaml').write_text(yaml.safe_dump(c));t.write_text('event,epoch,metrics_json\n')
                    _,d,t=paths(*self.items[1])
                    if marker=='csv_final':t.write_text('event,epoch,metrics_json\nfinal,20,{}\n')
                    else:(d/marker).write_text('partial or complete checkpoint')
                    with self.assertRaisesRegex(RuntimeError,'Possibly completed'):recovery.preserve_interrupted()
                    self.assertTrue((root/'foundation').is_dir());self.assertFalse(list(root.glob('*.interrupted_*')))

    def test_active_process_bad_config_or_completed_hash_fail_before_moves(self):
        d,t=self.partial('foundation','agiqa3k')
        with patch.object(recovery,'active_jobs',return_value=[{'pid':123}]),self.assertRaisesRegex(RuntimeError,'active'):recovery.preserve_interrupted()
        self.assertTrue(d.exists())
        (d/'launch_config.yaml').write_text('foreign: config')
        with self.assertRaisesRegex(ValueError,'Unrecognized'):recovery.preserve_interrupted()
        c,_,_=self.paths('foundation','agiqa3k');(d/'launch_config.yaml').write_text(yaml.safe_dump(c))
        (d/'completion_manifest.json').write_text('{}')
        with patch.object(recovery.protocol,'verify_foundation',side_effect=ValueError('hash mismatch')),self.assertRaises(ValueError):recovery.preserve_interrupted()
        self.assertTrue(d.exists());self.assertFalse(list(self.root.glob('*.interrupted_*')))

    def test_proc_guard_matches_script_argument_not_shell_command_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for pid,args in [(1,[b'python',b'scripts/run_protected_local.py',b'direct']),
                             (2,[b'python',b'scripts/train_protected_local.py']),
                             (3,[b'bash',b'-lc',b'echo train_cm_iqa.py']),
                             (4,[b'python',b'scripts/recover_protected_local.py'])]:
                d=root/str(pid);d.mkdir();(d/'cmdline').write_bytes(b'\0'.join(args)+b'\0')
            # Bypass the setUp stub to exercise the real implementation.
            # Use captured original function, without reloading protocol globals.
            found=ORIGINAL_ACTIVE_JOBS(root)
            self.assertEqual({p['pid'] for p in found},{1,2})

class BackgroundLauncherTests(unittest.TestCase):
    def prepare(self,root):
        (root/'scripts').mkdir();bin_dir=root/'bin';bin_dir.mkdir()
        shutil.copy(recovery.protocol.ROOT/'scripts/restart_protected_local_background.sh',root/'scripts/restart_protected_local_background.sh')
        fake_python=bin_dir/'fake_python';fake_python.write_text('#!/bin/sh\nexit "${FAKE_RECOVERY_STATUS:-0}"\n');fake_python.chmod(0o755)
        fake_nohup=bin_dir/'nohup';fake_nohup.write_text('#!/bin/sh\nprintf "gpu:%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n');fake_nohup.chmod(0o755)
        env=dict(os.environ,ABLATION_PYTHON=str(fake_python),PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
        env.pop('PROTECTED_DIRECT_GPU',None);env.pop('PROTECTED_WARM_GPU',None)
        return env

    def test_launcher_detaches_two_queues_with_distinct_logs_and_pids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.prepare(root)
            result=subprocess.run(['bash','scripts/restart_protected_local_background.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            logs=next((root/'run_logs').glob('protected_restart_*'))
            for _ in range(100):
                if all((logs/n).is_file() and (logs/n).stat().st_size for n in ('direct.log','warm_chain.log')):break
                time.sleep(.01)
            a=(logs/'direct.log').read_text();b=(logs/'warm_chain.log').read_text()
            self.assertIn('gpu:0',a);self.assertIn('direct',a);self.assertIn('setsid',a)
            self.assertIn('gpu:1',b);self.assertIn('warm-chain',b)
            self.assertTrue((logs/'direct.pid').read_text().strip().isdigit())
            self.assertTrue((logs/'warm_chain.pid').read_text().strip().isdigit())

    def test_failed_recovery_does_not_launch_either_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.prepare(root);env['FAKE_RECOVERY_STATUS']='1'
            result=subprocess.run(['bash','scripts/restart_protected_local_background.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'run_logs').exists())

ORIGINAL_ACTIVE_JOBS=recovery.active_jobs
if __name__=='__main__':unittest.main()
