"""Synthetic complete-state transfer, frozen outputs, local warmup and pipeline recovery."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from models.two_stage_models import load_pretrained_backbone
from utils.protected_local import (ProtectedLocalIQA,FULL_FORMAT,LOCAL_FORMAT,ALLOWED,
    save_selected_downstream,restore_foundation,transfer_warm_projection,frozen_digest,
    validate_snapshot_config,write_json_new,sha256)
from scripts import run_protected_local as protocol
from scripts.train_protected_local import fit,CleanRichHFLoader
from test_align_representation import TinyVisual
import test_align_representation as routes
import test_learnable_text_context as contexts
from test_frozen_visual_encoder_mode import LocalBN
from test_downstream_pipeline_controls import batch
from train_cm_iqa import build_optimizer,evaluate


def make(c):
    with patch('models.two_stage_models.GlobalVisualEncoder',TinyVisual),patch('models.two_stage_models.ResNet50LocalEncoder',LocalBN),patch('models.two_stage_models.FrozenCLIPTextEncoder',side_effect=lambda **kw:contexts.LearnableTextContextTests().make_encoder(5)):
        return ProtectedLocalIQA(c)


def config(v='old'):
    c=routes.config();c.update(seed=42,split_seed=42,scheduler={'min_lr':1e-6},dataset=protocol.reference_config('agiqa3k')['dataset']);c['train']=copy.deepcopy(protocol.config('warmup','richhf')['train'] if v=='warmup' else protocol.config(v,'agiqa3k')['train'])
    c['train']['epochs']=2;c['train']['amp']=False
    if v!='old':c['model']['align_region_word']={'version':'clip_region_word_v1'}
    return c


class StateTests(unittest.TestCase):
    def setUp(self):
        rng=torch.random.get_rng_state();self.addCleanup(torch.random.set_rng_state,rng);torch.manual_seed(42)

    def test_full_snapshot_transfer_preserves_all_heads_context_and_outputs_and_reports_new_weights(self):
        c=config();c['train'].update(save_selected_downstream=FULL_FORMAT,selection_metric='val_quality_SROCC',epochs=60)
        with tempfile.TemporaryDirectory() as directory:
            c.update(output_dir=directory,run_id='mature')
            old=make(c).eval()
            old.backbone.global_encoder.model_mean.fill_(.17)
            with torch.no_grad():
                for n,p in old.named_parameters():
                    if n.startswith(('align_global_head.','global_quality_head.')):p.add_(.13)
            final=dict(last_epoch=60,best_epoch=13,selection_metric='val_quality_SROCC',best_val={},best_test={})
            record=save_selected_downstream(old,c,final,{'sha256':'original'})
            ck=torch.load(Path(directory)/record['checkpoint'],weights_only=False)
            self.assertTrue(ck['training_complete']);self.assertEqual(set(ck['model']),set(old.state_dict()))
            newcfg=copy.deepcopy(c);newcfg['model']['align_region_word']={'version':'clip_region_word_v1'};new=make(newcfg)
            restore=restore_foundation(new,ck,newcfg,expected_run_id='mature',expected_pretrain_sha='original')
            self.assertEqual(len(restore['new_parameters']),2)
            for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)
            new.protect('scratch');x=torch.rand(4,3,16,16);prompts=['a','b','c','d']
            with torch.no_grad():
                a=old.forward_single(x,prompts);b=new.forward_single(x,prompts)
            for k in ('s_quality','s_quality_global','s_quality_patch','s_align_global'):
                torch.testing.assert_close(a[k],b[k],rtol=1e-6,atol=1e-7,msg=k)
            with self.assertRaises(FileExistsError):save_selected_downstream(old,c,final,{'sha256':'original'})
            for change in ('key','buffer','source','run','seed','complete','mode'):
                bad=copy.deepcopy(ck)
                if change=='key':bad['model'].pop('align_global_head.net.0.weight',None);bad['model'].pop(next(k for k in bad['model'] if k.startswith('align_global_head.')))
                elif change=='buffer':bad['nonpersistent_buffers'].pop('backbone.global_encoder.model_mean')
                elif change=='source':bad['upstream']['sha256']='foreign'
                elif change=='run':bad['run_id']='foreign'
                elif change=='seed':bad['cfg']['seed']=43
                elif change=='complete':bad['training_complete']=False
                else:bad['mode']='legacy'
                with self.assertRaises(ValueError):restore_foundation(new,bad,newcfg,expected_run_id='mature',expected_pretrain_sha='original')
        self.assertIsNone(save_selected_downstream(old,{'train':{}},{},None))

    def test_warm_projection_transfer_is_exact_and_does_not_replace_mature_score_head(self):
        c=config('warmup');pre=make(c);pre.protect('warmup')
        with torch.no_grad():
            for p in pre.align_region_word.parameters():p.add_(.25)
        state={k:v.clone() for k,v in pre.state_dict().items() if k.startswith(ALLOWED['warmup'])}
        ck=dict(format=LOCAL_FORMAT,mode='projection_only_v1',variant='warmup',training_complete=True,
            upstream={'sha256':'source'},frozen_before='same',frozen_after='same',model=state)
        student=make(config('warm'));head=copy.deepcopy(student.align_patch_head.state_dict())
        report=transfer_warm_projection(student,ck,'source')
        for k,v in pre.align_region_word.state_dict().items():torch.testing.assert_close(v,student.align_region_word.state_dict()[k],rtol=0,atol=0)
        for k,v in head.items():torch.testing.assert_close(v,student.align_patch_head.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(report['explicitly_not_transferred']);self.assertEqual(len(report['loaded']),2)
        for mutate in ('source','missing','freeze'):
            bad=copy.deepcopy(ck)
            if mutate=='source':bad['upstream']['sha256']='other'
            elif mutate=='missing':bad['model'].pop(next(k for k in state if k.startswith('align_region_word.')))
            else:bad['frozen_after']='changed'
            with self.assertRaises((ValueError,RuntimeError)):transfer_warm_projection(student,bad,'source')

    def test_protected_fit_updates_only_local_state_preserves_frozen_bn_and_calls_test_once(self):
        for variant in ('old','scratch','warmup'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                c=config(variant);c.update(output_dir=directory,results_csv=directory+'/metrics.csv',run_id='toy')
                m=make(c);m.protect(variant);before=frozen_digest(m)
                trainable={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
                self.assertTrue(all(n.startswith(ALLOWED[variant]) for n in trainable))
                self.assertFalse(m.backbone.training);self.assertFalse(m.quality_adapter.training)
                self.assertFalse(m.backbone.text_encoder.context_vectors.requires_grad)
                data=batch(['a','b','c','d']);data['image']=torch.rand(4,3,16,16)
                test=None if variant=='warmup' else [data]
                with contextlib.redirect_stdout(io.StringIO()),patch('scripts.train_protected_local.evaluate',wraps=evaluate) as evaluations:
                    final=fit(m,([data],[data],test),c,{'sha256':'upstream'},{},torch.device('cpu'))
                self.assertEqual(before,frozen_digest(m));self.assertEqual(final['frozen_before'],final['frozen_after'])
                self.assertTrue(any(not torch.equal(p,dict(m.named_parameters())[n]) for n,p in trainable.items()))
                self.assertEqual(sum(call.kwargs.get('split')=='test' for call in evaluations.call_args_list),0 if variant=='warmup' else 1)
                saved=torch.load(Path(directory)/'best_local.pth',weights_only=False)
                self.assertTrue(all(k.startswith(ALLOWED[variant]) for k in saved['model']))
                self.assertTrue(saved['training_complete'])
                if variant=='warmup':
                    out=m.forward_single(data['image'],data['prompt']);torch.testing.assert_close(out['s_align'],out['s_align_patch'])
                meta=json.loads((Path(directory)/'run_metadata.json').read_text())
                self.assertTrue(meta['checkpoint_verified_after_load']);self.assertEqual(meta['optimizer_initialization'][0]['lr'],1e-4)

    def test_clean_view_mapping_and_configuration_rejections(self):
        raw=dict(image_clean=torch.ones(2,3,16,16),prompt=['a','b'],quality_score=torch.ones(2),align_score=torch.zeros(2))
        loader=type('L',(),{'dataset':'data','__len__':lambda self:1,'__iter__':lambda self:iter([raw])})()
        adapted=next(iter(CleanRichHFLoader(loader)));self.assertIs(adapted['image'],raw['image_clean'])
        self.assertTrue(torch.equal(adapted['align_score'],torch.zeros(2))) # genuine zero labels retained
        raw.pop('image_clean')
        with self.assertRaises(ValueError):next(iter(CleanRichHFLoader(loader)))
        c=protocol.foundation_config('agiqa3k');c['train']['compare_selection_policies']=True
        with self.assertRaises(ValueError):validate_snapshot_config(c)


class ConfigTests(unittest.TestCase):
    def test_nine_run_plan_and_budget_sources_and_defaults(self):
        plan=protocol.plan('all',protocol.DATASETS);self.assertEqual(len(plan),9);self.assertEqual(len(set(plan)),9)
        self.assertEqual(len(protocol.plan('direct',protocol.DATASETS)),6)
        for ds in protocol.DATASETS:
            original=protocol.reference_config(ds);foundation=protocol.foundation_config(ds)
            expected=copy.deepcopy(original)
            for k in ('run_id','output_dir','results_csv'):expected[k]=foundation[k]
            expected['train'].update(save_selected_downstream=FULL_FORMAT,record_split_manifest=True,record_optimizer_initialization=True)
            self.assertEqual(expected,foundation)
            for v in protocol.VARIANTS:
                c=protocol.config(v,ds);self.assertEqual(c['dataset'],original['dataset']);self.assertEqual(c['train']['epochs'],20)
                self.assertEqual(c['train']['selection_metric'],'val_align_SROCC');self.assertEqual(c['train']['lambda_quality'],0)
                self.assertEqual(c['train']['pretrained_load_scope'],FULL_FORMAT)
                self.assertEqual(c['train']['layerwise_lr'],original['train']['layerwise_lr'])
        warm=protocol.config('warmup','richhf')
        original=yaml.safe_load((protocol.ROOT/'configs/projection_only_v1_baseline/pretrain.yaml').read_text())
        self.assertEqual(warm['dataset'],original['dataset']);self.assertEqual(warm['train']['human_supervision_view'],'three_view')
        self.assertEqual(warm['train']['lambda_align_aux'],0)
        for v,ds in [('bad','agiqa3k'),('old','richhf'),('warmup','agiqa3k'),('scratch','agiqa1k')]:
            with self.assertRaises(ValueError):protocol.config(v,ds)

    def test_cpu_refusal_and_existing_partial_never_retrained(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(protocol,'paths',return_value=(Path(directory)/'new',Path(directory)/'new.csv')),patch.object(protocol,'dependencies',return_value={'sha256':'source'}),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):protocol.run('scratch','agiqa3k')
            (Path(directory)/'new').mkdir()
            with self.assertRaises(FileExistsError):protocol.run('scratch','agiqa3k')
            with patch.object(protocol,'verify',side_effect=ValueError('incomplete')),self.assertRaises(ValueError):protocol.run('scratch','agiqa3k',True)

class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        for directory in ('configs/current_tuning_reference','configs/projection_only_v1_baseline'):
            shutil.copytree(protocol.ROOT/directory,self.root/directory)
        (self.root/'scripts').mkdir();shutil.copy(protocol.ROOT/'scripts/run_align_tuning.py',self.root/'scripts/run_align_tuning.py')
        (self.root/'docs').mkdir()
        for name in ('current_tuning_reference.json','current_tuning_reference.md','protected_local.md'):
            shutil.copy(protocol.ROOT/'docs'/name,self.root/'docs'/name)
        self.source=protocol.original.reference()['source']
        for target,attr,value in ((protocol,'ROOT',self.root),(protocol,'source',lambda:self.source)):
            patcher=patch.object(target,attr,value);patcher.start();self.addCleanup(patcher.stop)
        self.reference={'val_quality_SROCC':1.,'val_quality_PLCC':.9,'val_align_global_SROCC':.7,'val_align_global_PLCC':.8}
        self.test={'test_quality_SROCC':1.,'test_quality_PLCC':.9,'test_align_global_SROCC':.7,'test_align_global_PLCC':.8,'test_align_SROCC':.8}
        self.split=dict(indices={'train':[0],'val':[1],'test':[2]},dataset_ids_sha256='dataset',train_seed=42,split_seed=42)

    def foundation(self,ds):
        import test_align_tuning as fixture
        r=protocol.foundation_runner();helper=fixture.RunnerTests();helper.source=self.source
        with patch.object(fixture,'r',r):
            d,t=helper.fake('foundation',ds,False)
            with t.open() as f:rows=list(csv.DictReader(f))
            for row in rows:
                if row['event']=='best_test':row['metrics_json']=json.dumps(self.test)
                if row['event']=='final':
                    value=json.loads(row['metrics_json']);value.update(best_val=self.reference,best_test=self.test);row['metrics_json']=json.dumps(value)
            with t.open('w') as f:writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            # SHA-backed synthetic full checkpoint. Actual tensor completeness
            # and strict restoration are exercised by StateTests above.
            torch.save({'model':{'fake':torch.ones(1)},'format':FULL_FORMAT},d/'best_selected_downstream.pth')
            write_json_new(d/'selected_checkpoint_manifest.json',dict(format=FULL_FORMAT,checkpoint='best_selected_downstream.pth',sha256=sha256(d/'best_selected_downstream.pth'),
                run_id=protocol.foundation_config(ds)['run_id'],selected_epoch=40,last_epoch=60,upstream_sha256=self.source['sha256']))
            helper.seal(d,t)
        protocol.verify_foundation(ds)
        self.split=json.loads((d/'data_split_manifest.json').read_text())

    def artifact(self,v,ds):
        c=protocol.config(v,ds);d,t=protocol.paths(v,ds);d.mkdir(parents=True)
        for name in ('launch_config.yaml','config_snapshot.yaml'):(d/name).write_text(yaml.safe_dump(c))
        dep=protocol.dependencies(v,ds)
        write_json_new(d/'launch_manifest.json',dict(version=protocol.VERSION,variant=v,dataset=ds,dependencies=dep,config_sha256=sha256(d/'launch_config.yaml')))
        (d/'console.log').write_text('synthetic complete')
        model={prefix+'weight':torch.ones(1) for prefix in ALLOWED[v] if prefix!='align_region_word.'}
        if v!='old':model.update({'align_region_word.image_projection.weight':torch.zeros(256,1024),'align_region_word.text_projection.weight':torch.zeros(256,768)})
        initialization={}
        if v=='warm':initialization['warmup_transfer']={'loaded':[k for k in model if k.startswith('align_region_word.')],'explicitly_not_transferred':['align_patch_head.weight']}
        meta=dict(variant=v,seed=42,split_seed=42,checkpoint=dep,checkpoint_verified_after_load=True,frozen_before='same',
            trainable_parameters=list(model),optimizer_initialization=[dict(name='align',lr=1e-4,weight_decay=.002)],initialization=initialization)
        write_json_new(d/'initial_metadata.json',meta);write_json_new(d/'run_metadata.json',dict(meta,frozen_after='same',training_complete=True))
        best_val=dict(self.reference,val_align_SROCC=.8)
        final=dict(last_epoch=20,best_epoch=12,selection_metric='val_align_SROCC',best_selection_value=.8,
            best_val=best_val,best_test={} if v=='warmup' else self.test,frozen_before='same',frozen_after='same')
        with t.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['event','epoch','metrics_json']);w.writeheader()
            w.writerow(dict(event='run_start',epoch=0,metrics_json='{}'))
            w.writerow(dict(event='reference_validation',epoch=0,metrics_json=json.dumps(dict(self.reference,val_align_SROCC=.5))))
            for epoch in range(1,21):
                value=.8-abs(epoch-12)/100;m={'val_'+k:x for k,x in self.reference.items()};m['val_val_align_SROCC']=value
                w.writerow(dict(event='epoch',epoch=epoch,metrics_json=json.dumps(m)))
            w.writerow(dict(event='final',epoch=12,metrics_json=json.dumps(final)))
        ck=dict(format=LOCAL_FORMAT,mode='projection_only_v1',variant=v,training_complete=True,run_id=c['run_id'],cfg=c,
                upstream=dep,initialization=initialization,model=model,**final)
        torch.save(ck,d/'best_local.pth')
        write_json_new(d/'local_checkpoint_manifest.json',dict(format=LOCAL_FORMAT,variant=v,checkpoint='best_local.pth',
            sha256=sha256(d/'best_local.pth'),selected_epoch=12,last_epoch=20,upstream=dep))
        if v=='warmup':write_json_new(d/'official_split_manifest.json',dict(identities={'train':['train'],'val':['val'],'test':['test']},counts={'train':1,'val':1,'test':1},csv_sha256={}))
        else:write_json_new(d/'data_split_manifest.json',self.split)
        self.seal(v,ds)
        return d,t

    def seal(self,v,ds):
        d,t=protocol.paths(v,ds)
        (d/'completion_manifest.json').write_text(json.dumps(dict(files_sha256={n:sha256(d/n) for n in protocol.local_files(v)},csv_sha256=sha256(t))))

    def test_all_nine_complete_sources_and_pack_excludes_weights(self):
        for ds in protocol.DATASETS:self.foundation(ds)
        self.artifact('warmup','richhf');protocol.verify('warmup','richhf')
        for ds in protocol.DATASETS:
            for v in protocol.VARIANTS:
                self.artifact(v,ds);protocol.verify(v,ds)
        logs=self.root/'run_logs';logs.mkdir()
        journal=logs/'protected_local_recovery_test.json'
        journal.write_text(json.dumps(dict(format='protected_local_recovery_v1',version=protocol.VERSION,upstream=self.source,complete=True)))
        with contextlib.redirect_stdout(io.StringIO()):protocol.pack()
        import tarfile
        with tarfile.open(self.root/f'{protocol.VERSION}_results.tar.gz') as tar:
            names=tar.getnames();self.assertEqual(sum(n.endswith('.csv') for n in names),9)
            self.assertFalse(any(n.endswith('.pth') for n in names));self.assertIn('summary.json',names);self.assertIn('run_logs/protected_local_recovery_test.json',names)
        with self.assertRaises(FileExistsError):protocol.pack()

    def test_changed_source_frozen_state_and_missing_projection_fail_even_after_resealing(self):
        self.foundation('agiqa3k');self.artifact('warmup','richhf')
        for v,kind in (('old','freeze'),('scratch','projection'),('warm','source')):
            d,t=self.artifact(v,'agiqa3k')
            if kind=='freeze':
                meta=json.loads((d/'run_metadata.json').read_text());meta['frozen_after']='changed';(d/'run_metadata.json').write_text(json.dumps(meta))
            elif kind=='projection':
                ck=torch.load(d/'best_local.pth',weights_only=False);ck['model'].pop('align_region_word.text_projection.weight');torch.save(ck,d/'best_local.pth')
                m=json.loads((d/'local_checkpoint_manifest.json').read_text());m['sha256']=sha256(d/'best_local.pth');(d/'local_checkpoint_manifest.json').write_text(json.dumps(m))
            else:
                launch=json.loads((d/'launch_manifest.json').read_text());launch['dependencies']['sha256']='foreign';(d/'launch_manifest.json').write_text(json.dumps(launch))
            self.seal(v,'agiqa3k')
            with self.assertRaises(ValueError):protocol.verify(v,'agiqa3k')

class WarmChainTests(unittest.TestCase):
    def test_plan_and_order_pretrain_then_wait_then_warm_without_foundation_launch(self):
        self.assertEqual(protocol.plan('warm-chain',protocol.DATASETS),[
            ('warmup','richhf'),('warm','agiqa3k'),('warm','aigciqa2023')])
        events=[]
        with patch.object(protocol,'run',side_effect=lambda *args:events.append(('run',args))),patch.object(protocol,'wait_for_foundations',side_effect=lambda *args:events.append(('wait',args))):
            protocol.run_warm_chain(protocol.DATASETS,True,24)
        self.assertEqual(events,[('run',('warmup','richhf',True)),('wait',(protocol.DATASETS,24)),
                                 ('run',('warm','agiqa3k',True)),('run',('warm','aigciqa2023',True))])
        with patch.object(protocol,'run',side_effect=RuntimeError('warmup failed')),patch.object(protocol,'wait_for_foundations') as wait,self.assertRaises(RuntimeError):
            protocol.run_warm_chain(protocol.DATASETS)
        wait.assert_not_called()

    def test_wait_handles_late_completion_and_verifies_each_source_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);now=[0.];slept=[]
            def path(ds):return root/ds/'best_selected_downstream.pth'
            def sleep(seconds):
                slept.append(seconds);now[0]+=seconds
                ds=protocol.DATASETS[len(slept)-1];path(ds).parent.mkdir(exist_ok=True)
                (path(ds).parent/'completion_manifest.json').write_text('{}')
            path(protocol.DATASETS[0]).parent.mkdir()
            (path(protocol.DATASETS[0]).parent/'completion_manifest.json').write_text('')
            with patch.object(protocol,'foundation_path',side_effect=path),patch.object(protocol,'verify_foundation') as verify,patch.object(protocol.time,'monotonic',side_effect=lambda:now[0]),patch.object(protocol.time,'sleep',side_effect=sleep),contextlib.redirect_stdout(io.StringIO()):
                protocol.wait_for_foundations(protocol.DATASETS,1,30)
            self.assertEqual(slept,[30,30]);self.assertEqual([c.args for c in verify.call_args_list],[(ds,) for ds in protocol.DATASETS])

    def test_corrupt_completion_and_timeout_never_launch_warm_or_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'best_selected_downstream.pth';marker=path.parent/'completion_manifest.json';marker.write_text('corrupt')
            with patch.object(protocol,'foundation_path',return_value=path),patch.object(protocol,'verify_foundation',side_effect=ValueError('bad source')),patch.object(protocol.time,'sleep') as sleep,self.assertRaises(ValueError):
                protocol.wait_for_foundations(('agiqa3k',))
            sleep.assert_not_called();self.assertEqual(marker.read_text(),'corrupt');marker.unlink()
            now=[0.]
            with patch.object(protocol,'foundation_path',return_value=path),patch.object(protocol.time,'monotonic',side_effect=lambda:now[0]),patch.object(protocol.time,'sleep',side_effect=lambda s:now.__setitem__(0,now[0]+s)),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(TimeoutError):
                protocol.wait_for_foundations(('agiqa3k',),1/3600,30)
        with patch.object(protocol,'run') as run,patch.object(protocol,'wait_for_foundations',side_effect=TimeoutError('not ready')),self.assertRaises(TimeoutError):
            protocol.run_warm_chain(protocol.DATASETS,True)
        self.assertEqual(run.call_count,1);self.assertEqual(run.call_args.args,('warmup','richhf',True))

if __name__=='__main__':unittest.main()
