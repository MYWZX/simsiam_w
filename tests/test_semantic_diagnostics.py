"""Frozen-forward semantic diagnostics and opt-in complete selected snapshots, CPU only."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from utils import semantic_snapshot as snap,semantic_diagnostics as diag
from utils.selection_policies import SelectionComparison
from train_cm_iqa import get_selection_value
from scripts import alignment_semantic_diagnostics as cli
import test_alignment_evidence as fixture
from test_align_representation import config as tiny


class Net(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(2,1);self.register_buffer('offset',torch.tensor(1.),persistent=False)
    def forward(self,x):return self.linear(x)+self.offset


def config(root):
    return {'run_id':'synthetic','output_dir':str(root),'train':{'epochs':60,'selection_metric':'val_quality_SROCC',
            'selection_mode':'max',snap.FLAG:snap.FORMAT},'model':{'fusion':{'mode':'projection_only_v1'}}}


class SnapshotTests(unittest.TestCase):
    def test_default_off_and_strict_full_state_including_nonpersistent_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config(root);m=Net().eval();x=torch.randn(3,2);before=m(x).detach()
            result={'best_epoch':5,'selection_metric':'val_quality_SROCC','best_val':{'val_quality_SROCC':.8},'best_test':{}}
            state=torch.random.get_rng_state()
            snap.save(c,'quality',result,60,{'sha256':'source'},m.state_dict(),snap.buffers(m))
            torch.testing.assert_close(state,torch.random.get_rng_state(),atol=0,rtol=0)
            payload=torch.load(root/'semantic_quality.pth',weights_only=True);target=Net();snap.load_strict(target,payload)
            torch.testing.assert_close(target(x),before,atol=0,rtol=0)
            self.assertFalse(target.training);self.assertTrue(all(not p.requires_grad for p in target.parameters()))
            with self.assertRaises(FileExistsError):snap.save(c,'quality',result,60,{'sha256':'source'},m.state_dict(),snap.buffers(m))
            for change in ('missing','dtype','buffers','pretrain'):
                bad=copy.deepcopy(payload)
                if change=='missing':bad['model'].pop('linear.bias')
                elif change=='dtype':bad['model']['linear.bias']=bad['model']['linear.bias'].double()
                elif change=='buffers':bad['nonpersistent_buffers']={}
                else:bad['format']='pretrain'
                with self.assertRaises(ValueError):snap.load_strict(Net(),bad)
            c['train'].pop(snap.FLAG)
            self.assertIsNone(snap.save(c,'quality',{},0,None,{},{}))

    def test_selected_buffers_and_tensors_are_immutable_per_policy(self):
        policies={'quality':'val_quality_SROCC','joint':'val_quality_align_mean_SROCC'}
        tracker=SelectionComparison(policies,capture_nonpersistent=True);m=Net()
        for ep,q,a in [(1,.8,.8),(2,.9,.4)]:
            with torch.no_grad():m.offset.fill_(ep);m.linear.weight.fill_(ep)
            tracker.update(m,{'val_quality_SROCC':q,'val_align_SROCC':a,'val_align_N':3},ep,get_selection_value)
        with torch.no_grad():m.offset.fill_(99);m.linear.weight.fill_(99)
        observed=[]
        def evaluate():
            observed.append(float(m.offset));return ({'offset':float(m.offset)}, {})
        results=tracker.evaluate_selected(m,evaluate)
        self.assertEqual(observed,[2.,1.]);self.assertEqual(results['quality']['best_epoch'],2)
        self.assertEqual(float(tracker.best['quality']['state']['linear.weight'][0,0]),2.)
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);c['train'].update(compare_selection_policies=True,selection_comparison_version='quality_mean_v1')
            snap.export_comparison(m,c,tracker,results,60,{'sha256':'source'})
            self.assertTrue((Path(tmp)/'semantic_joint.pth').is_file())
        old=SelectionComparison(policies);old.update(m,{'val_quality_SROCC':.8,'val_align_SROCC':.8,'val_align_N':3},1,get_selection_value)
        self.assertNotIn('buffers',old.best['quality'])

    def test_invalid_export_budget_or_policy_rejected_before_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);c['train']['epochs']=20
            with self.assertRaises(ValueError):snap.enabled(c)
            c=config(tmp);c['train']['compare_selection_policies']=True
            with self.assertRaises(ValueError):snap.enabled(c)
            with self.assertRaises(ValueError):snap.save(config(tmp),'quality',{'best_epoch':1,'selection_metric':'val_quality_SROCC'},14,{'sha256':'s'},{},{})
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_real_main_export_hooks_keep_both_policies_and_complete_selected_buffers(self):
        import train_cm_iqa as training
        from contextlib import ExitStack
        for compare in (False,True):
            with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp);source=root/'source.pth';source.write_bytes(b'fixture source')
                c=config(root);c.update(seed=42,split_seed=42,dataset={'type':'agiqa3k_simsiam'})
                c['train'].update(pretrained_checkpoint=str(source),amp=False,early_stop_patience=60)
                if compare:c['train'].update(compare_selection_policies=True,selection_comparison_version='quality_mean_v1')
                stack.enter_context(patch('sys.argv',['train_cm_iqa.py','--config','unused','--output-dir',str(root),'--results-csv',str(root/'results.csv')]))
                stack.enter_context(patch.object(training,'load_yaml',return_value=c))
                stack.enter_context(patch.object(training,'build_dataloaders',return_value=('train','val','test')))
                m=Net();stack.enter_context(patch.object(training,'build_iqa_model',return_value=m))
                stack.enter_context(patch.object(training,'load_pretrained_backbone'))
                stack.enter_context(patch.object(training,'build_optimizer',side_effect=lambda c,m:torch.optim.SGD(m.parameters(),lr=.1)))
                stack.enter_context(patch.object(training,'apply_progressive_finetune_stage',return_value=('all',{'all':3})))
                stack.enter_context(patch.object(training,'format_metrics',return_value='fixture'))
                stack.enter_context(patch('torch.cuda.is_available',return_value=False))
                count=[]
                def train(model,loader,optimizer,scaler,device,cfg,epoch,**kw):
                    count.append(epoch)
                    with torch.no_grad():model.offset.fill_(epoch);model.linear.weight.fill_(epoch)
                    return {'loss':0.}
                def evaluate(model,loader,device,cfg,split):
                    if split=='test':return {'test_quality_SROCC':.5}
                    return {'val_quality_SROCC':.8 if float(model.offset)==1 else .9,
                            'val_align_SROCC':.8 if float(model.offset)==1 else .4,'val_align_N':3}
                stack.enter_context(patch.object(training,'train_one_epoch',side_effect=train))
                stack.enter_context(patch.object(training,'evaluate',side_effect=evaluate))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()));training.main()
                self.assertEqual(len(count),60)
                quality=torch.load(root/'semantic_quality.pth',weights_only=True)
                self.assertEqual(quality['epoch'],2);self.assertEqual(float(quality['nonpersistent_buffers']['offset']),2.)
                if compare:
                    joint=torch.load(root/'semantic_joint.pth',weights_only=True)
                    self.assertEqual(joint['epoch'],1);self.assertEqual(float(joint['model']['linear.weight'][0,0]),1.)


class ReviewTests(unittest.TestCase):
    def test_replacements_are_single_candidate_edits_not_labels(self):
        edits=diag.replacements('Two red cats sit left of a dog.')
        self.assertTrue({'color','count','spatial','entity'}.issubset({x['category'] for x in edits}))
        self.assertEqual(next(x for x in edits if x['category']=='color')['edited_prompt'],'Two blue cats sit left of a dog.')
        self.assertTrue(all('review_status' not in x for x in edits))

    def test_only_confirmed_relative_labels_enter_ordering_accuracy(self):
        rows=[]
        for status,delta in [('candidate',-99),('confirmed_original_better',.2),('confirmed_equivalent',.01)]:
            rows.append(dict(model='m',category='color',review_status=status,delta_global=delta,delta_local=delta,delta_final=delta))
        report=diag.summarize(rows)['m']['all'];self.assertEqual(report['confirmed_order_N'],1)
        self.assertEqual(report['final']['ordering_accuracy'],1);self.assertEqual(report['candidate_N'],1)
        self.assertIsNone(diag.summarize(rows[:1])['m']['all']['final']['ordering_accuracy'])

    def test_review_note_required_and_nonvalidation_rejected(self):
        d={'version':'semantic_pairs_v1','dataset':'agiqa3k','split':'val','cases':[dict(case_id='a',category='color',
            original_prompt='red',edited_prompt='blue',review_status='confirmed_original_better',reviewer='',review_note='')]}
        with self.assertRaises(ValueError):diag.validate_cases(d)
        d['cases'][0].update(reviewer='human',review_note='Image depicts red rather than blue');diag.validate_cases(d)
        d['split']='test'
        with self.assertRaises(ValueError):diag.validate_cases(d)

    def test_truncated_prompt_is_refused_before_any_forward(self):
        encoder=SimpleNamespace(num_context_tokens=5,model=SimpleNamespace(context_length=8),
            tokenizer=lambda texts:torch.tensor([[1,2,3,99,0,0,0,0]]))
        m=SimpleNamespace(backbone=SimpleNamespace(text_encoder=encoder))
        with self.assertRaisesRegex(ValueError,'truncated'):diag.infer(m,torch.zeros(1,3,16,16),'too long')

    def test_missing_selected_weights_never_fall_back_to_pretraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'last_pretrain.pth').write_bytes(b'not downstream')
            spec={'run_dir':str(d),'run_id':'x'}
            with patch.object(diag,'run_evidence',return_value=({}, {'best_epoch':3},{'checkpoint':{'sha256':'x'}},None)):
                with self.assertRaisesRegex(FileNotFoundError,'pretraining'):diag.snapshot_evidence(d,spec)

    def test_seed_alone_is_not_accepted_as_recorded_membership(self):
        dataset=SimpleNamespace();loaders=[SimpleNamespace(dataset=SimpleNamespace(dataset=dataset))]*3
        actual={'indices':{'train':[0],'val':[1],'test':[2]},'dataset_ids_sha256':'ids','test_ids_sha256':'t'}
        with tempfile.TemporaryDirectory() as tmp,patch('datasets.build_dataloaders',return_value=loaders),patch('utils.refit_protocol.split_manifest',return_value=actual):
            with self.assertRaisesRegex(ValueError,'not recorded'):diag.validation_cohort({'loader':{}},tmp,{})


class FrozenIntegrationTests(unittest.TestCase):
    def test_two_full_snapshots_infer_without_training_and_render_real_heatmaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image.png';Image.fromarray(np.full((16,16,3),100,dtype=np.uint8)).save(image)
            records={};items=[]
            for role in diag.ROLES:
                folder=root/role;folder.mkdir();c=tiny();c.update(run_id=role,output_dir=str(folder),seed=42,split_seed=42,dataset={'input_size':16})
                c['train'].update(epochs=60,selection_metric='val_quality_SROCC',selection_mode='max',export_semantic_snapshots=snap.FORMAT)
                if role=='region_tau01':
                    c['model']['align_region_word']={'version':'clip_region_word_v1'}
                    c['model']['align_evidence']={'version':'align_evidence_v1','temperature':{'mode':'fixed','initial':.1}}
                m=fixture.model(c).eval();result={'best_epoch':3,'selection_metric':'val_quality_SROCC','best_val':{},'best_test':{}}
                snap.save(c,'quality',result,60,{'sha256':'source'},m.state_dict(),snap.buffers(m))
                spec=dict(dataset='agiqa3k',role=role,run_id=role,run_dir=str(folder),label=role);items.append(spec)
                records[role]=dict(spec=spec,cfg=c,selected=result,metadata={'checkpoint':{'sha256':'source'}},
                    checkpoint=str(folder/'semantic_quality.pth'),sha256=diag.digest(folder/'semantic_quality.pth'),manifest=str(folder/'semantic_quality.json'),table='synthetic')
            base=dict(row_index=1,image_path=str(image),image_sha256=diag.digest(image),original_prompt='a',reviewer='human',review_note='Synthetic relative-label fixture')
            cases=[dict(base,case_id='1_entity',category='entity',edited_prompt='b',review_status='confirmed_original_better'),
                   dict(base,case_id='1_color',category='color',edited_prompt='c',review_status='candidate')]
            document=dict(version='semantic_pairs_v1',dataset='agiqa3k',split='val',cases=cases)
            def build(c,p):
                m=fixture.model(c);m.backbone.text_encoder.tokenizer.decode=lambda ids:'content';return m
            with patch.object(diag,'snapshot_evidence',side_effect=lambda root,s:copy.deepcopy(records[s['role']])),patch.object(cli,'validate_cohort_cases',return_value='synthetic'),patch.object(diag,'validate_config'),patch('train_cm_iqa.build_iqa_model',side_effect=build),patch('train_cm_iqa.train_one_epoch',side_effect=AssertionError('no training')),patch('torch.optim.AdamW',side_effect=AssertionError('no optimizer')):
                cli.run(items,document,root/'results','cpu',True,1)
            report=diag.read_json(root/'results/report.json')
            self.assertFalse(report['training_started'])
            self.assertEqual(report['input_cases_sha256'],diag.digest(root/'results/input_cases.json'))
            self.assertTrue((root/'results/model_provenance.json').is_file())
            for role in diag.ROLES:
                self.assertEqual(report['summary'][role]['all']['confirmed_order_N'],1)
                self.assertEqual(report['summary'][role]['all']['candidate_N'],1)
            self.assertEqual(len(list((root/'results/figures').glob('*.png'))),2)
            metadata=[diag.read_json(p) for p in (root/'results/maps').glob('*.json')]
            self.assertTrue(any(m['original_scope']=='whole_prompt_only_no_token_map' for m in metadata))
            with patch('torch.load',side_effect=AssertionError('packing cannot load weights')):cli.pack(root/'results',root/'results.tar.gz')
            import tarfile
            with tarfile.open(root/'results.tar.gz') as tar:self.assertFalse(any(x.endswith(('.pth','.pt','.bin')) for x in tar.getnames()))

    def test_capture_plan_only_writes_configs_and_new_explicit_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);items=diag.default_specs(cli.ROOT)
            with patch('subprocess.Popen',side_effect=AssertionError('no training')):cli.capture_plan(items,root/'plan')
            data=diag.read_json(root/'plan/models.json');self.assertEqual(len(data['models']),4)
            self.assertTrue(all(x['label'].startswith('new_capture_') for x in data['models']))
            self.assertFalse(diag.read_json(root/'plan/plan.json')['training_started'])

if __name__=='__main__':unittest.main()
