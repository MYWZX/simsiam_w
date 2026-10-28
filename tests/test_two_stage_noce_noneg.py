"""Negative-free pretraining preserves positive supervision and refit routing."""
import copy
from pathlib import Path
import subprocess
import os
import unittest
from unittest.mock import patch
import torch
import yaml
from train_cm_pretrain import _compute_richhf_relational_losses

ROOT=Path(__file__).resolve().parents[1]

class PositiveOnlyTests(unittest.TestCase):
    def fixture(self):
        class Projector:
            def __init__(self):self.patch_calls=[];self.text_calls=[]
            def project_patch_regions(self,features,weights):
                self.patch_calls.append(weights)
                z=(features*weights.unsqueeze(-1)).sum(1)/(weights.sum(1,keepdim=True)+1e-6)
                return {'p':z,'z_proj':z}
            def project_aux_texts(self,phrases,device):
                self.text_calls.append(phrases)
                z=torch.tensor([[1.,2.],[2.,1.]],requires_grad=True)
                return {'p':z,'z_proj':z}
        patches=torch.randn(2,4,2,requires_grad=True)
        out={'img1_z':torch.zeros(2,2),'align_patch_features_clean':patches,
             'align_prompt_z':torch.ones(2,2),'align_prompt_p':torch.ones(2,2),
             'answer_z':torch.ones(2,2),'answer_p':torch.ones(2,2)}
        spatial=torch.tensor([[[[0.,1.],[0.,1.]]]]).repeat(2,1,1,1)
        batch={'misalignment_map1':spatial,'misalignment_map2':spatial,
               'misalignment_map_clean':spatial,'misalignment_map_valid':torch.ones(2),
               'prompt_word_labels':['[1,0]','[1,0]'],'answer_confidence':torch.ones(2)}
        cfg={'human_supervision_view':'three_view','lambda_img_prompt_local':.05,
             'lambda_prompt_answer_word':.05,'disable_negative_relations':True}
        return Projector(),out,batch,cfg,patches

    def test_no_negative_projection_or_push_positive_gradients_remain(self):
        model,out,batch,cfg,patches=self.fixture()
        with patch('train_cm_pretrain._simsiam_push_apart',side_effect=AssertionError('negative push executed')),patch('train_cm_pretrain._simsiam_relative_rank',side_effect=AssertionError('negative rank executed')):
            local,word=_compute_richhf_relational_losses(model,out,batch,['good bad']*2,torch.device('cpu'),cfg)
        self.assertEqual(len(model.patch_calls),1)
        self.assertEqual(model.text_calls,[['good','good']])
        self.assertTrue(torch.isfinite(local+word))
        local.backward();self.assertGreater(patches.grad.abs().sum().item(),0.)

    def test_positive_coefficient_is_not_doubled(self):
        model,out,batch,cfg,_=self.fixture()
        with patch('train_cm_pretrain.simsiam_pair_loss',return_value=torch.tensor(4.)):
            local,word=_compute_richhf_relational_losses(model,out,batch,['good bad']*2,torch.device('cpu'),cfg)
        self.assertEqual(local.item(),2.);self.assertEqual(word.item(),2.)

    def test_legacy_mode_still_projects_and_pushes_negatives(self):
        model,out,batch,cfg,_=self.fixture();cfg.pop('disable_negative_relations')
        with patch('train_cm_pretrain._simsiam_push_apart',return_value=torch.tensor(0.)) as push:
            _compute_richhf_relational_losses(model,out,batch,['good bad']*2,torch.device('cpu'),cfg)
        self.assertEqual(push.call_count,2);self.assertEqual(len(model.patch_calls),2)
        self.assertEqual(model.text_calls,[['good','good'],['bad','bad']])

    def test_relative_mode_cannot_reintroduce_negatives_when_disabled(self):
        model,out,batch,cfg,_=self.fixture();cfg['local_relation_loss']='relative_rank'
        with patch('train_cm_pretrain._simsiam_relative_rank',side_effect=AssertionError('negative rank executed')):
            _compute_richhf_relational_losses(model,out,batch,['good bad']*2,torch.device('cpu'),cfg)
        self.assertEqual(len(model.patch_calls),1)

class PipelineTests(unittest.TestCase):
    def test_configs_use_new_source_and_keep_other_losses(self):
        p=ROOT/'configs/protocol_v2'
        pre=yaml.safe_load((p/'pretrain_both_noce_noneg.yaml').read_text())
        old=yaml.safe_load((p/'pretrain.yaml').read_text())
        expected=copy.deepcopy(old)
        expected['model']['quality_text_bank']['enabled']=False
        expected['train'].update(lambda_quality_text_ce=0.,lambda_quality_text_anchor=0.,disable_negative_relations=True,lambda_prompt_answer_gap=0.)
        for k in ['run_id','output_dir','results_csv']:expected[k]=pre[k]
        self.assertEqual(pre,expected)
        for ds in ['agiqa1k','agiqa3k','aigciqa2023']:
            c=yaml.safe_load((p/f'{ds}_both_noce_noneg.yaml').read_text())
            self.assertEqual(c['train']['pretrained_checkpoint'],pre['output_dir']+'/best_component_anchor_pretrain.pth')
            self.assertFalse(c['model']['quality_text_bank']['enabled'])
            self.assertEqual(c['train']['lambda_neg'],0.)
            self.assertEqual(c['train']['lambda_quality_text_ce'],0.)
            self.assertGreater(c['train']['lambda_srocc_quality'],0.)
            self.assertIn('_both_noce_noneg_',c['output_dir'])

    def test_new_pretrain_dev_refit_routes(self):
        env={**os.environ,'ABLATION_PYTHON':'/bin/echo'}
        for stage,expected in [('pretrain','pretrain_both_noce_noneg.yaml'),('dev','agiqa1k_both_noce_noneg.yaml'),('refit','--refit-plan outputs/protocolv2_plans/agiqa1k_both_noce_noneg.json')]:
            result=subprocess.check_output(['bash','scripts/run_protocol_v2.sh',stage,'both_noce_noneg','agiqa1k'],cwd=ROOT,env=env,text=True)
            self.assertIn(expected,result)

if __name__=='__main__':unittest.main()
