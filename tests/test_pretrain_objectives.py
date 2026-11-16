"""Image-Prompt weight sweep protocol tests; fake artifacts only, no real training."""
import contextlib
import copy
import io
import csv
import json
import tarfile
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import yaml
from scripts import run_pretrain_objectives as sweep
import test_pretrain_component_strength as component_tests


class PretrainObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.fixture=component_tests.ComponentStrengthTests('test_incomplete_refuses_pack')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        shutil.copytree(sweep.ROOT/sweep.CONFIG,self.root/sweep.CONFIG)
        p=patch.object(sweep,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def test_exact_single_factor_and_original_baseline_downstreams(self):
        paths=[]
        for variant,(path,value) in sweep.VARIANTS.items():
            r=sweep.make_runner(variant);cfg=r.validate('pretrain');expected=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain'))
            for k in ('run_id','output_dir','results_csv'):expected[k]=cfg[k]
            target=expected['train']
            for key in path[:-1]:target=target[key]
            target[path[-1]]=value
            self.assertEqual(cfg,expected)
            self.assertFalse(any(k in cfg['train'] for k in ('resume','pretrained_checkpoint','refit_plan')))
            self.assertEqual(cfg['model']['nss_text_gate']['nss_dim'],6)
            self.assertEqual(cfg['train']['lambda_img_siam'],.15);self.assertEqual(cfg['train']['lambda_img_prompt'],.25)
            self.assertEqual(cfg['train']['lambda_artifact_anchor'],.01);self.assertEqual(cfg['train']['lambda_artifact_ordinal'],.015)
            self.assertEqual(cfg['train']['lambda_quality_text_ce'],0.)
            for ds in sweep.STAGES[1:]:
                actual=r.validate(ds);expected=yaml.safe_load((self.root/f'configs/current_tuning_reference/{ds}.yaml').read_text())
                for k in ('run_id','output_dir','results_csv'):expected[k]=actual[k]
                expected['train']['pretrained_checkpoint']=cfg['output_dir']+'/best_component_anchor_pretrain.pth'
                self.assertEqual(actual,expected)
            paths.append(str(r.paths('pretrain')[0]))
        self.assertEqual(len(set(paths)),10)
        self.assertEqual(set(sweep.QUEUES['low'])|set(sweep.QUEUES['high']),set(sweep.VARIANTS))
        self.assertFalse(set(sweep.QUEUES['low'])&set(sweep.QUEUES['high']))
        with self.assertRaises(ValueError):sweep.make_runner('unknown')

    def test_dry_run_and_cpu_no_outputs(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch.object(sweep.pipeline,'run_stage',side_effect=AssertionError('no trainer')),contextlib.redirect_stdout(io.StringIO()):
                sweep.pipeline.run_pipeline(r,'all',True)
            with patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'CUDA'):r.run('pretrain')
            self.assertFalse(r.paths('pretrain')[0].exists())
            with self.assertRaises(FileNotFoundError):r.run('agiqa3k',True)

    def test_completed_source_tamper_and_pack(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                r.run('pretrain');source=r.completed('pretrain');r.pack('pretrain')
            self.assertEqual(source['version'],f'projection_only_v1_ctx5_obj_{variant}')
            self.assertTrue((self.root/f'{r.VERSION}_pretrain_results.tar.gz').exists())
            with self.assertRaises(FileExistsError):r.run('pretrain')
            d,_=r.paths('pretrain');(d/'best_component_anchor_pretrain.pth').write_bytes(b'wrong-source')
            with self.assertRaises(ValueError):r.run('agiqa3k',True)

    def test_changed_protocol_rejected(self):
        r=sweep.make_runner('answer010');p=r.CONFIG/'pretrain.yaml'
        cfg=yaml.safe_load(p.read_text());cfg['train']['lambda_img_siam']=.10
        p.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):r.validate('pretrain')


    def test_thirty_downstreams_verify_sources_splits_budget_and_pack(self):
        from utils.agiqa1k_subset1078 import expected_resplit_manifest
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',self.fixture.fake_trainer(r)),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                r.run('pretrain')
            source=r.completed('pretrain')
            outer=self
            class Downstream:
                def __init__(inner,command,**kwargs):
                    outer.assertEqual(command[1],'train_cm_iqa.py')
                    stage=Path(command[3]).stem
                    cfg=r.validate(stage);d,table=r.paths(stage)
                    (d/'config_snapshot.yaml').write_text(yaml.safe_dump(cfg))
                    r.write_new(d/'run_metadata.json',dict(seed=42,split_seed=cfg['split_seed'],checkpoint_verified_after_load=True,checkpoint={'sha256':source['sha256']}))
                    if cfg['train'].get('record_split_manifest',False):
                        indices=expected_resplit_manifest(3407)['split_indices'] if stage=='agiqa1k' else {}
                        r.write_new(d/'data_split_manifest.json',dict(train_seed=42,split_seed=cfg['split_seed'],indices=indices))
                    with table.open('w') as f:
                        writer=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);writer.writeheader()
                        for epoch in range(1,61):writer.writerow(dict(event='epoch',epoch=epoch,metrics_json='{}'))
                        writer.writerow(dict(event='final',epoch=60,metrics_json=json.dumps(dict(last_epoch=60,selection_metric='val_quality_SROCC'))))
                    inner.stdout=iter(['fake downstream completed\n'])
                def wait(inner):return 0
            with patch('torch.cuda.is_available',return_value=True),patch.object(r.subprocess,'Popen',Downstream),patch.object(r.subprocess,'run'),patch('utils.run_metadata.collect_run_metadata',return_value={'fake':True}),contextlib.redirect_stdout(io.StringIO()):
                for stage in sweep.STAGES[1:]:r.run(stage)
                r.pack('downstream')
                for stage in sweep.STAGES[1:]:
                    self.assertEqual(r.completed(stage),source)
                    self.assertEqual(r.read_json(r.paths(stage)[0]/'source_fingerprint.json'),source)
                    with self.assertRaises(FileExistsError):r.run(stage)
            with tarfile.open(self.root/f'{r.VERSION}_downstream_results.tar.gz') as tar:
                names=tar.getnames()
                self.assertEqual(sum(n.endswith('.csv') for n in names),4)
                self.assertEqual(sum(n.endswith('completion_manifest.json') for n in names),4)
                self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in names))
            self.assertFalse((r.paths('agiqa3k')[0]/'data_split_manifest.json').exists())
            # A subsequent file change must invalidate both completion and packaging.
            metadata=r.paths('agiqa1k')[0]/'run_metadata.json'
            data=r.read_json(metadata);data['checkpoint']['sha256']='0'*64
            metadata.write_text(json.dumps(data))
            with self.assertRaises(ValueError):r.completed('agiqa1k')

    def test_pipeline_order_failure_stops_and_outputs_are_preserved(self):
        for variant in sweep.VARIANTS:
            r=sweep.make_runner(variant)
            with patch.object(sweep.pipeline,'run_stage') as launch:
                sweep.pipeline.run_pipeline(r,'all')
                self.assertEqual([c.args[1] for c in launch.call_args_list],list(sweep.STAGES))
            with patch.object(sweep.pipeline,'run_stage',side_effect=RuntimeError('pretrain failed')) as launch:
                with self.assertRaises(RuntimeError):sweep.pipeline.run_pipeline(r,'all')
                self.assertEqual(launch.call_count,1)
            d,table=r.paths('aigciqa2023');table.parent.mkdir(exist_ok=True);table.write_text('keep')
            with patch.object(sweep.pipeline,'run_stage') as launch:
                with self.assertRaises(FileExistsError):sweep.pipeline.run_pipeline(r,'all')
                launch.assert_not_called()
            self.assertEqual(table.read_text(),'keep')


class QueueTests(unittest.TestCase):
    def test_preflight_all_five_before_launch_failure_stops_and_dry_run_never_launches(self):
        for queue in sweep.QUEUES:
            fake_runners=[object() for _ in sweep.QUEUES[queue]]
            with patch.object(sweep,'make_runner',side_effect=fake_runners),patch.object(sweep.pipeline,'run_pipeline') as preflight,patch.object(sweep,'dispatch') as launch:
                sweep.run_queue(queue,'all')
                self.assertEqual([c.args for c in preflight.call_args_list],[(r,'continue',True) for r in fake_runners])
                self.assertEqual([c.args for c in launch.call_args_list],[(r,'all') for r in fake_runners])
            with patch.object(sweep,'make_runner',side_effect=fake_runners),patch.object(sweep.pipeline,'run_pipeline',side_effect=[None,None,None,None,ValueError('last config invalid')]),patch.object(sweep,'dispatch') as launch:
                with self.assertRaises(ValueError):sweep.run_queue(queue,'all')
                launch.assert_not_called()
            with patch.object(sweep,'make_runner',side_effect=fake_runners),patch.object(sweep.pipeline,'run_pipeline'),patch.object(sweep,'dispatch',side_effect=RuntimeError('train failed')) as launch:
                with self.assertRaises(RuntimeError):sweep.run_queue(queue,'all')
                self.assertEqual(launch.call_count,1)
            with patch.object(sweep,'make_runner',side_effect=fake_runners),patch.object(sweep.pipeline,'run_pipeline'),patch.object(sweep,'dispatch') as launch:
                sweep.run_queue(queue,'all',True);launch.assert_not_called()
        with self.assertRaises(ValueError):sweep.run_queue('low','agiqa3k')


class ActualObjectiveTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.saved_rng=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,self.saved_rng)
        torch.manual_seed(42)

    def test_relation_coefficients_reach_real_evaluation_loss_independently(self):
        import torch
        from train_cm_pretrain import evaluate
        from test_dual_quality_representation import config,batch
        from test_projection_ablations import make
        c=config();c['train'].update(lambda_img_prompt_local=.05,lambda_prompt_answer_word=.05)
        m=make(c);data=batch()
        # The map/word estimator is isolated here; the actual evaluation loss
        # aggregator and I-A/P-A losses run unmodified on the tiny real model.
        with patch('train_cm_pretrain._compute_richhf_relational_losses',return_value=(torch.tensor(.7),torch.tensor(.9))):
            baseline=evaluate(m,[data],torch.device('cpu'),c)
            names={'lambda_img_answer':'img_answer','lambda_prompt_answer':'prompt_answer',
                   'lambda_img_prompt_local':'img_prompt_local','lambda_prompt_answer_word':'prompt_answer_word'}
            for variant,(path,value) in sweep.VARIANTS.items():
                if len(path)!=1:continue
                changed=copy.deepcopy(c);changed['train'][path[0]]=value
                observed=evaluate(m,[data],torch.device('cpu'),changed)
                expected=(value-c['train'][path[0]])*baseline[names[path[0]]]
                self.assertAlmostEqual(observed['loss']-baseline['loss'],expected,places=6)
                for raw in names.values():self.assertEqual(observed[raw],baseline[raw])

    def test_actual_local_word_estimators_remain_positive_only(self):
        import torch
        import test_two_stage_noce_noneg as positive_tests
        from train_cm_pretrain import _compute_richhf_relational_losses
        fixture=positive_tests.PositiveOnlyTests()
        model,out,data,cfg,patches=fixture.fixture()
        base=_compute_richhf_relational_losses(model,out,data,['good bad']*2,torch.device('cpu'),cfg)
        for key in ('lambda_img_prompt_local','lambda_prompt_answer_word'):
            for value in (.025,.1):
                changed=copy.deepcopy(cfg);changed[key]=value
                with patch('train_cm_pretrain._simsiam_push_apart',side_effect=AssertionError('negative push executed')),patch('train_cm_pretrain._simsiam_relative_rank',side_effect=AssertionError('negative ranking executed')):
                    local,word=_compute_richhf_relational_losses(model,out,data,['good bad']*2,torch.device('cpu'),changed)
                torch.testing.assert_close(local,base[0],atol=0,rtol=0);torch.testing.assert_close(word,base[1],atol=0,rtol=0)
                gradient=torch.autograd.grad(value*local+value*word,patches,retain_graph=True)[0]
                self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(gradient.abs().sum(),0)

    def test_anchor_temperature_changes_both_components_and_has_finite_gradients(self):
        import torch
        from train_cm_pretrain import _compute_quality_component_anchor_ordinal_losses as component
        from train_cm_pretrain import _compute_quality_anchor_ordinal_losses as one
        tcfg=copy.deepcopy(sweep.pipeline.baseline.validate('pretrain')['train'])
        q=torch.randn(12,256,requires_grad=True);out={'quality_clean_z':q,'img1_z':q}
        data={'artifact_score':torch.linspace(.3,.95,12),'aesthetics_score':torch.linspace(.35,.9,12)}
        weight=torch.ones(12);base_loss,base_pred=component(out,data,weight,tcfg)
        for tau in (.1,.3):
            changed=copy.deepcopy(tcfg);changed['quality_component_anchor_ordinal']['anchor_temperature']=tau
            with patch('train_cm_pretrain._compute_quality_anchor_ordinal_losses',wraps=one) as call:
                losses,pred=component(out,data,weight,changed)
            self.assertEqual([c.kwargs['anchor_cfg']['anchor_temperature'] for c in call.call_args_list],[tau,tau])
            for comp in ('artifact','aesthetics'):
                self.assertFalse(torch.equal(pred[comp+'_anchor_score'],base_pred[comp+'_anchor_score']))
                self.assertFalse(torch.equal(losses[comp+'_anchor_ce'],base_loss[comp+'_anchor_ce']))
            total=sum(.01*losses[c+'_anchor_ce']+.015*losses[c+'_ordinal_bce'] for c in ('artifact','aesthetics'))
            grad=torch.autograd.grad(total,q)[0];self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum(),0)


if __name__=='__main__':unittest.main()
