"""Refined fixed-temperature protocol, old-control reuse, isolation and packaging."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import os
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from models.region_word_alignment import RegionWordAlignment
from models.alignment_evidence import spatial_diagnostics
from scripts import run_align_fixed_temperature as old,run_align_fixed_temperature_fine as study
import test_align_fixed_temperature as helpers
import test_align_tuning as fixture
ROOT=Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_single_factor_source_and_old_defaults_are_preserved_across_factories(self):
        reference=old.make_runner();r=study.make_runner();seen=[]
        for ds in r.DATASETS:
            control=reference.config('tau010',ds)
            for v,tau in study.VARIANTS.items():
                c=r.config(v,ds);seen.append(c['output_dir']);expected=copy.deepcopy(control)
                for k in ('run_id','output_dir','results_csv'):expected[k]=c[k]
                expected['model']['align_evidence']['temperature']['initial']=tau
                self.assertEqual(c,expected)
                self.assertNotEqual(c['output_dir'],control['output_dir'])
        self.assertEqual(len(set(seen)),4)
        self.assertEqual(reference.config('tau010','agiqa3k')['model']['align_evidence']['temperature']['initial'],.1)
        self.assertEqual(old.make_runner().VERSION,old.VERSION)
        self.assertEqual(reference.VARIANTS,{'tau100':1.,'tau010':.1})
        self.assertEqual(r.VARIANTS,{'tau005':.05,'tau020':.2})
        for v,ds in [('tau010','agiqa3k'),('tau005','agiqa1k')]:
            with self.assertRaises(ValueError):r.config(v,ds)

    def test_all_three_temperatures_apply_to_softmax_and_sharpen_monotonically(self):
        m=RegionWordAlignment(2,2,2)
        with torch.no_grad():m.image_projection.weight.copy_(torch.eye(2));m.text_projection.weight.copy_(torch.eye(2))
        entropy=[]
        for tau in (.05,.1,.2):
            _,_,w,_,_=m(torch.eye(2)[None],torch.tensor([[[1.,0.]]]),torch.tensor([[True]]),tau,{'mode':'mean'})
            torch.testing.assert_close(w[0,:,0],torch.tensor([1.,0.]).div(tau).softmax(0))
            entropy.append(spatial_diagnostics(w)['spatial_entropy'].item())
        self.assertLess(entropy[0],entropy[1]);self.assertLess(entropy[1],entropy[2])


class ProtocolTests(unittest.TestCase):
    write_rows=helpers.ArtifactTests.write_rows
    def setUp(self):
        self.r=study.make_runner();p=patch.object(fixture,'r',self.r);p.start();self.addCleanup(p.stop)
        self.f=fixture.RunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        shutil.copy(ROOT/'docs/align_fixed_temperature_fine.md',self.f.root/'docs')

    def artifact(self,v,ds):return helpers.ArtifactTests.artifact(self,v,ds)

    def control(self,ds):
        previous=study.previous_runner(self.r.ROOT)
        proxy=SimpleNamespace(r=previous,f=self.f,write_rows=self.write_rows)
        with patch.object(fixture,'r',previous):return helpers.ArtifactTests.artifact(proxy,'tau010',ds)

    def test_four_new_runs_and_two_old_controls_verified_and_packed_without_changes_or_weights(self):
        controls={}
        for ds in self.r.DATASETS:
            d,t=self.control(ds);controls[t]=t.read_bytes()
            for v in study.VARIANTS:
                self.artifact(v,ds);self.r.verify(v,ds,None,self.f.source)
        with contextlib.redirect_stdout(io.StringIO()):self.r.pack()
        with tarfile.open(self.f.root/(study.VERSION+'_results.tar.gz')) as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),6)
            self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
            record=json.load(tar.extractfile('fixed_temperature_fine_protocol.json'))
            self.assertFalse(record['reference_retrained']);self.assertEqual(record['reference_temperature'],.1)
        for t,before in controls.items():self.assertEqual(t.read_bytes(),before)

    def test_missing_old_control_refuses_queue_before_any_new_training(self):
        with patch.object(self.r,'run') as launch,self.assertRaisesRegex(RuntimeError,'Incomplete'):
            self.r.run_all('all',continue_completed=True)
        launch.assert_not_called()

    def test_control_parameter_initialization_and_source_tampering_refused(self):
        cd,ct=self.control('agiqa3k');d,t=self.artifact('tau005','agiqa3k')
        self.r.verify('tau005','agiqa3k',None,self.f.source)
        meta=self.r.base.read_json(d/'run_metadata.json');meta['region_word_initialization'][0]['sha256']='c'*64
        (d/'run_metadata.json').write_text(json.dumps(meta));self.f.seal(d,t)
        with self.assertRaisesRegex(ValueError,'fingerprints'):self.r.verify('tau005','agiqa3k',None,self.f.source)
        (cd/'source_fingerprint.json').write_text(json.dumps({'sha256':'wrong'}))
        with self.assertRaises(ValueError):self.r.control('agiqa3k',None,self.f.source)

    def test_verified_control_skipped_and_only_new_cases_queued(self):
        for ds in self.r.DATASETS:self.control(ds)
        with patch.object(self.r,'run') as launch:self.r.run_all('all',continue_completed=True)
        self.assertEqual([c.args for c in launch.call_args_list],[(v,ds) for ds in self.r.DATASETS for v in study.VARIANTS])

    def test_plot_uses_existing_point_one_csv_without_rewriting_it(self):
        from scripts import plot_align_fixed_temperature_fine as plotting
        cd,ct=self.control('agiqa3k');before=ct.read_bytes()
        for v in study.VARIANTS:self.artifact(v,'agiqa3k')
        with patch.object(plotting,'make_runner',return_value=self.r):plotting.render('agiqa3k',self.f.root/'fine')
        self.assertEqual(ct.read_bytes(),before)
        self.assertGreater((self.f.root/'fine.png').stat().st_size,10000)
        self.assertTrue((self.f.root/'fine.pdf').read_bytes().startswith(b'%PDF'))

    def test_fine_process_guard_separates_parent_and_old_study(self):
        from scripts.preflight_align_fixed_temperature_fine import active_jobs
        p=self.f.root/'proc';p.mkdir()
        jobs=[(1,['bash','scripts/launch_align_fixed_temperature_fine_background.sh']),
              (2,['python','scripts/run_align_fixed_temperature_fine.py','low']),
              (3,['python','scripts/run_align_fixed_temperature.py','unit']),
              (4,['python','train_cm_iqa.py','--config','/outputs/align_fixed_temperature_fine_v1/run.yaml'])]
        for pid,args in jobs:
            d=p/str(pid);d.mkdir();(d/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args))
        self.assertEqual(set(active_jobs(p)),{'2','4'})


class LauncherTests(unittest.TestCase):
    def test_preflight_failure_never_starts_fine_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir();shutil.copy(ROOT/'scripts/launch_align_fixed_temperature_fine_background.sh',root/'scripts')
            script=root/'python';script.write_text('#!/bin/bash\n[[ "$*" != *preflight_align_fixed_temperature_fine* ]]\n');script.chmod(0o755)
            env=dict(os.environ,ABLATION_PYTHON=str(script))
            result=subprocess.run(['bash','scripts/launch_align_fixed_temperature_fine_background.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'run_logs').exists())

if __name__=='__main__':unittest.main()
