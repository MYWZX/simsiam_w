import sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'tests')]
import torch
from test_quality_residual_feature import build
from models.fusion import SemanticStructureFusion
m=build().eval();m.quality_global_only=False
names=['quality_adapter','global_quality_head','nss_proj','nss_gate','patch_quality_head','local_patch_proj','text_to_fused','align_global_adapter','align_text_adapter','align_patch_adapter','align_patch_img_proj','align_patch_txt_proj','align_global_head','align_patch_head','align_gate']
inputs={k:[] for k in names};outputs={k:[] for k in names}
for name in names:
 def hook(mod,inp,out,name=name):
  inputs[name].append(inp[0].detach().clone());outputs[name].append(out.detach().clone())
 getattr(m,name).register_forward_hook(hook)
x=torch.randn(3,3,8,8)
with torch.no_grad():o=m.forward_single(x,prompts=['cat']*3)
checks=[]
def check(label,a,b):
 torch.testing.assert_close(a,b);checks.append(label)
z=o['z'];p0=inputs['align_patch_adapter'][0];pq=inputs['patch_quality_head'][0];n=m._compute_local_nss_features(x,o['f_loc'])
check('z enters Quality Adapter',inputs['quality_adapter'][0],z)
a=torch.nn.functional.normalize(outputs['quality_adapter'][0],dim=-1)
r,alpha=m._quality_residual_feature(z,a)
check('Global Quality head consumes residual r, not bare a',inputs['global_quality_head'][0],r)
check('NSS computed from image; Pq=P0+N(NSS)',pq,p0+outputs['nss_proj'][0])
check('NSS gate consumes concat(Pq,NSS)',inputs['nss_gate'][0],torch.cat([pq,n],-1))
check('Quality local scores consume Pq',inputs['patch_quality_head'][0],pq)
check('Same L called twice with same raw feature map',inputs['local_patch_proj'][0],inputs['local_patch_proj'][1])
check('z enters image Alignment adapter',inputs['align_global_adapter'][0],z)
check('U output enters text Alignment adapter',inputs['align_text_adapter'][0],outputs['text_to_fused'][0])
v=outputs['align_global_adapter'][0];u=outputs['align_text_adapter'][0];ap=outputs['align_patch_adapter'][0]
check('Global pair head uses concat(v,u,abs(v-u))',inputs['align_global_head'][0],torch.cat([v,u,(v-u).abs()],-1))
check('Image matching projection consumes AP(P0)',inputs['align_patch_img_proj'][0],ap)
check('Text matching projection consumes u',inputs['align_patch_txt_proj'][0],u)
check('Attention pools AP values, not XI projected vectors',inputs['align_patch_head'][0][:,:8],(o['align_patch_weights'].unsqueeze(-1)*ap).sum(1))
e=inputs['align_patch_head'][0][:,:8]
check('Local pair head uses concat(e,u,abs(e-u))',inputs['align_patch_head'][0],torch.cat([e,u,(e-u).abs()],-1))
check('Alignment gate conditions on v,u,Ag,Al',inputs['align_gate'][0],torch.cat([v,u,o['s_align_global'][:,None],o['s_align_patch'][:,None]],-1))
check('Scalar beta Quality score fusion',o['s_quality'],o['quality_mix']*o['s_quality_patch']+(1-o['quality_mix'])*o['s_quality_global'])
check('Sample-wise gamma Alignment score fusion',o['s_align'],o['align_gate']*o['s_align_global']+(1-o['align_gate'])*o['s_align_patch'])
f=SemanticStructureFusion(semantic_dim=12,local_dim=10,embed_dim=8,num_heads=2,dropout=0).eval()
g=torch.randn(3,12);fl=torch.randn(3,10,2,2)
with torch.no_grad():
 q=f.semantic_proj(g).unsqueeze(1);kv=f.local_proj(fl).flatten(2).transpose(1,2);att,_=f.attn(q,kv,kv,need_weights=False);h=f.norm1(q+att);expected=f.norm2(h+f.ffn(h)).squeeze(1)
 check('Fusion: own local projection, attention, residuals, norms, FFN',f(g,fl),expected)
result={'status':'passed','checks':checks,'scope':'CPU tiny encoders, real model heads and actual SemanticStructureFusion; no GPU, dataset or performance claim'}
Path('docs/presentations/framework_route_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,indent=2))
