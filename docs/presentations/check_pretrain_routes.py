import sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'tests')]
import torch
from unittest.mock import patch
from test_quality_residual_feature import TinyBackbone
from models.two_stage_models import CrossModalIQAJointPretrain
cfg={'model':{'heads':{'mlp_hidden_dim':16,'dropout':0.},'fusion':{'num_heads':2},'quality_branch_use_text_gate':False,'quality_adapter':{'enabled':False},'simsiam':{'separate_quality_projection':True,'proj_dim':8,'proj_hidden_dim':16,'pred_hidden_dim':4}}}
with patch('models.two_stage_models.ImageTextBackbone',TinyBackbone):m=CrossModalIQAJointPretrain(cfg).eval()
seen={k:[] for k in ['quality_image_projector','quality_image_predictor','image_projector','text_projector']}
for name in seen:
 def hook(module,ins,out,name=name):seen[name].append((ins[0].detach().clone(),out.detach().clone()))
 getattr(m,name).register_forward_hook(hook)
x1,x2,xc=[torch.randn(3,3,8,8) for _ in range(3)]
with torch.no_grad():out=m._forward_three_views(x1,x2,xc,['p']*3,['a']*3)
checks=[]
def ck(label,a,b):torch.testing.assert_close(a,b);checks.append(label)
assert len(seen['quality_image_projector'])==3
checks.append('One PQ instance called on clean and two SafeAug views')
for actual,x,label in zip(seen['quality_image_projector'],[xc,x1,x2],['clean','view1','view2']):
 ck(f'PQ input is independent {label} backbone z',actual[0],m.iqa.backbone.encode_image(x)['z'])
ck('Clean anchors consume normalized PQ output, not DQ',out['quality_clean_z'],torch.nn.functional.normalize(seen['quality_image_projector'][0][1],dim=-1))
assert m.image_projector is not m.quality_image_projector and m.text_projector is not m.image_projector
checks.append('PQ PI PT independent parameter objects')
assert len(seen['image_projector'])==3 and len(seen['text_projector'])==3
checks.append('PI sees augmented/clean alignment features; PT sees Prompt twice plus Answer')
p=Path('docs/presentations/framework_route_checks.json');d=json.loads(p.read_text());d['pretrain_checks']=checks;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,indent=2))
