"""Render exact code-derived diagrams. Requires Graphviz `dot`, no Python packages."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

class Diagram:
    def __init__(self, name, title):
        self.name = name
        self.cluster_index = 0
        self.lines = ['digraph G {', 'graph [rankdir=LR, pack=true, packmode="array_u1", bgcolor="white", pad=0.3, nodesep=0.25, ranksep=0.55, fontname="DejaVu Sans", fontsize=21, labelloc=t, label='+json.dumps(title)+'];', 'node [shape=box, style="rounded,filled", fillcolor="#eef5fc", color="#38658a", fontname="DejaVu Sans", fontsize=11, margin="0.13,0.09"];', 'edge [color="#30485f", arrowsize=0.65, fontname="DejaVu Sans", fontsize=9];']
    def cluster(self, name, title, color):
        self.cluster_index += 1
        self.lines += ['subgraph cluster_'+name+' {', 'sortv='+str(100-self.cluster_index)+';', 'label='+json.dumps(title)+'; color="'+color+'"; fontsize=15; style=rounded;']
    def end(self): self.lines.append('}')
    def node(self, key, label, color=None):
        self.lines.append(key+' [sortv=0,label='+json.dumps(label)+((',fillcolor="'+color+'"') if color else '')+'];')
    def edge(self, a,b,label='',dashed=False):
        self.lines.append(a+' -> '+b+' [label='+json.dumps(label)+(',style=dashed,color="#9b6b22"' if dashed else '')+'];')
    def chain(self, *keys):
        for a,b in zip(keys,keys[1:]):self.edge(a,b)
    def save(self):
        self.lines.append('}')
        path=ROOT/(self.name+'.dot'); path.write_text(('\n'.join(self.lines)+'\n').replace('array_u1', 'array_u'+str(self.cluster_index+1)))
        for fmt in ['svg','png']:
            subprocess.run(['dot','-T'+fmt,str(path),'-o',str(ROOT/(self.name+'.'+fmt))],check=True)

TEAL='#dcf3ed'; PURPLE='#eee4fa'; AMBER='#fff0cb'; GRAY='#ededed'
p=Diagram('pretrain_heads_and_sharing','PRETRAINING | Actual forward heads and parameter sharing\nSame [ID] = same parameter object; different IDs = independent parameters')
p.cluster('backbone','A. Image and text inputs / shared backbone','#2c6399')
for key,text in [('views','SafeAug1 / SafeAug2 / Clean\nSeparate forwards through SAME V/R/F'),('v','[V] CLIP visual\n768-D global'),('r','[R] ResNet50\n256-D local map'),('f','[F] Fusion: attention + FFN\nGlobal query, local keys/values'),('z','z1 / z2 / zc: 256-D'),('lp','Local maps\nClean map supplies MOS / region paths'),('texts','Prompt / Answer / auxiliary phrases\nSAME trainable contexts [C]'),('t','[T] Frozen CLIP text Transformer'),('u','[U] text_to_fused\n768 -> 256')]:p.node(key,text)
p.chain('views','v','f','z');p.chain('views','r','f');p.edge('r','lp');p.chain('texts','t','u');p.end()
p.cluster('quality','B. Quality projection space | shared across three views','#1d8c78')
for key,text,col in [('qinputs','z1 / z2 / zc',TEAL),('pq','[PQ] quality_image_projector\n256 -> 512 -> 512 -> 256\nBN / ReLU / dropout / final BN',TEAL),('q12','q1 / q2 (normalized)',TEAL),('qc','qc (normalized Clean projection)',TEAL),('dq','[DQ] quality_image_predictor\n256 -> 128 -> 256',TEAL),('ii','I-I SimSiam loss\npredictor vs stop-gradient target',AMBER),('anchors','Artifact + Aesthetics\nfixed Anchor / Ordinal losses\nNO learned regression head',AMBER),('bqm','[BQ.M] image_to_text\nLinear 256 -> 768',TEAL),('qrefs','5 quality phrases + independent\nclass_context [5,4,768]\n-> SAME frozen [T] -> 5 references',TEAL),('ce','5-way similarity logits -> soft-label CE',AMBER),('qcp','[DQ] also runs on qc\nOutput computed, NOT used by Anchor/CE',GRAY)]:p.node(key,text,col)
p.chain('qinputs','pq');p.edge('pq','q12');p.edge('pq','qc');p.chain('q12','dq','ii');p.edge('q12','ii','stop-gradient targets',True);p.edge('qc','anchors');p.chain('qc','bqm','ce');p.edge('qrefs','ce');p.edge('qc','qcp');p.end()
p.cluster('semantic','C. Semantic projection heads | exact upstream routes','#8552aa')
for key,text in [('simg','zc for main I-P / I-A losses\nz1/z2 also computed'),('ag','[AG] align_global_adapter'),('pi','[PI] image_projector\nIndependent of PQ and PT'),('di','[DI] image_predictor'),('sp','Prompt: [T+C] -> [U]'),('at','[AT] align_text_adapter'),('xt','[XT] align_patch_txt_proj\n+ L2 normalize'),('sa','Answer: [T+C] -> [U]\nNo AT or XT'),('pt','[PT] text_projector\nSAME for Prompt / Answer / phrases\nIndependent of PQ and PI'),('dt','[DT] text_predictor\nSAME for all text relations'),('loss','I-P / I-A / P-A\nSymmetric SimSiam with stop-gradient targets')]:p.node(key,text,PURPLE if key!='loss' else AMBER)
p.chain('simg','ag','pi','di','loss');p.chain('sp','at','xt','pt','dt','loss');p.edge('sa','pt');p.edge('pi','loss','targets',True);p.edge('pt','loss','targets',True);p.end()
p.cluster('mos','D. Clean-view MOS scoring | does NOT pass through PQ','#cb8b27')
for key,text in [('mzin','zc'),('qg','[QG] global_quality_head\nMLP -> sigmoid'),('mlocal','Clean local map -> SAME [L] local_patch_proj\n+ [N] nss_proj(NSS) -> Pq'),('qp','[QP] patch_quality_head\npatch score'),('ng','[NG] nss_gate(concat Pq,NSS)\nsoftmax patch weights'),('qpool','Weighted sum -> patch Q'),('mix','Learned scalar beta\nMix global Q and patch Q'),('mos','Quality MOS regression / ranking'),('auxin','concat(zc, weighted Pq evidence, abs difference)'),('ha','[HA] Artifact auxiliary regression head'),('he','[HE] Aesthetics auxiliary regression head'),('ascore','Clean z / Prompt / local tokens\n-> AG / AT / AP + XI / XT\nSame Align scoring path as downstream'),('ahg','[AHG] align_global_head'),('ahp','[AHP] align_patch_head'),('gate','[G] align_gate\nper-sample score mixing'),('amos','Align MOS regression / ranking / soft-SROCC')]:p.node(key,text,AMBER)
p.chain('mzin','qg','mix','mos');p.chain('mlocal','qp','qpool','mix');p.chain('mlocal','ng','qpool');p.edge('mzin','auxin');p.edge('mlocal','auxin');p.edge('ng','auxin');p.edge('auxin','ha');p.edge('auxin','he');p.chain('ascore','ahg','gate','amos');p.edge('ascore','ahp');p.edge('ahp','gate');p.end()
p.cluster('local','E. Region / word feedback | shared heads, distinct upstream features','#8552aa')
for key,text in [('reg','Clean map -> [L] -> [AP] -> [XI] normalize\nHuman map-weighted region pooling'),('regpath','SAME [AG] -> [PI] -> [DI]'),('phrase','Aligned / misaligned phrases\n[T+C] -> [U] -> [AT]\nNO XT'),('phhead','SAME [PT] -> [DT]'),('pref','Main Prompt [XT] output\nSAME [PT] -> [DT] called again\nShared parameters, separate forward/BN/dropout'),('rel','Region / word pull-push losses\nTargets use stop-gradient')]:p.node(key,text,PURPLE)
p.chain('reg','regpath','rel');p.chain('phrase','phhead','rel');p.edge('pref','rel');p.end()
p.node('note','ACTIVE CONFIG: Quality Adapter OFF in pretraining; separate_quality_projection=True.\nMOS heads do not directly train PQ; Anchor/CE do. Shared backbone receives both tasks.\nPQ -> downstream QA is INITIALIZATION, not runtime sharing. PI/PT and predictors not transferred.\nExisting checkpoint PCGrad ON. Legacy Final score computed but not supervised.',GRAY)
p.save()

d=Diagram('downstream_heads_and_sharing','DOWNSTREAM | Actual scoring heads, shared modules and auxiliary path\nCurrent Large residual_feature, blend=0. Same [ID] = same parameter object')
d.cluster('feature','A. One shared feature backbone','#2c6399')
for key,text in [('img','Whole image 224x224'),('v','[V] CLIP ViT-L/14 visual\nTrainable, 768-D'),('r','[R] ResNet50\nTrainable local map'),('f','[F] SemanticStructureFusion'),('z','z: 256-D'),('l','[L] local_patch_proj\nConv1x1 256 -> 256'),('p0','P0: raw local patch tokens\nSAME L weights in Q/Align calls'),('p','Real Prompt + [C] shared contexts'),('t','[T] Frozen CLIP text Transformer'),('u','[U] text_to_fused 768 -> 256')]:d.node(key,text)
d.chain('img','v','f','z');d.chain('img','r','f');d.chain('r','l','p0');d.chain('p','t','u');d.end()
d.cluster('q','B. Quality scoring heads','#1d8c78')
for key,text,col in [('qz','Input z',TEAL),('qa','[QA] quality_adapter\nPQ weights initialize this MLP',TEAL),('a','a = normalized Adapter output',TEAL),('res','r = z + alpha ||z|| normalize(a)\nNorm detached; alpha learned, init 0.1',TEAL),('qg','[QG] global_quality_head\n256 -> 256 -> 128 -> 1 + sigmoid',TEAL),('qnss','Input image -> NSS6',TEAL),('n','[N] nss_proj 6 -> 256 -> 256',TEAL),('ql','Input P0 from SAME [L]',TEAL),('pqa','Pq = P0 + nss_proj(NSS)',TEAL),('qp','[QP] patch_quality_head\n256 -> 256 -> 1 + sigmoid',TEAL),('ng','[NG] nss_gate concat(Pq,NSS)\n262 -> 256 -> 1 -> softmax',TEAL),('pool','Weighted patch scores -> Qpatch',TEAL),('quality','Quality = beta Qpatch + (1-beta) Qglobal\nbeta = learned scalar sigmoid',TEAL),('diag','SAME [QG] also evaluates z and a\nDropout OFF for diagnostic calls\nNo separate quality_adapter_head',GRAY)]:d.node(key,text,col)
d.chain('qz','qa','a','res','qg','quality');d.edge('qz','res','z bypass');d.chain('qnss','n','pqa');d.edge('ql','pqa');d.chain('pqa','qp','pool','quality');d.chain('pqa','ng','pool');d.edge('qnss','ng','raw NSS');d.edge('qz','diag');d.edge('a','diag');d.end()
d.cluster('align','C. Align scoring heads | no NSS injection here','#8552aa')
for key,text in [('az','Input z'),('ag','[AG] align_global_adapter'),('at0','Input U(text)'),('at','[AT] align_text_adapter'),('ap0','Input P0 from SAME [L]'),('ap','[AP] align_patch_adapter'),('pairg','concat(v,u,abs(v-u))'),('ahg','[AHG] align_global_head\n768 -> 256 -> 128 -> 1 + sigmoid'),('xi','[XI] align_patch_img_proj\nLinear256 -> 256 + normalize'),('xt','[XT] align_patch_txt_proj\nLinear256 -> 256 + normalize'),('att','dot-product -> softmax weights'),('ev','weighted sum of AP output -> e'),('pairp','concat(e,u,abs(e-u))'),('ahp','[AHP] align_patch_head\n768 -> 256 -> 128 -> 1 + sigmoid'),('gin','concat(v,u,Aglobal,Apatch)'),('gate','[G] align_gate\n514 -> 256 -> 1 + sigmoid -> g'),('align','Align = g Aglobal + (1-g) Apatch')]:d.node(key,text,PURPLE)
d.chain('az','ag','pairg','ahg');d.chain('at0','at','pairg');d.chain('ap0','ap','xi','att','ev','pairp','ahp');d.chain('at','xt','att');d.edge('ap','ev','values');d.edge('at','pairp');
for key in ['ag','at','ahg','ahp']:d.edge(key,'gin')
d.chain('gin','gate','align');d.edge('ahg','align');d.edge('ahp','align');d.end()
d.cluster('bank','D. Auxiliary Quality text head | NOT a final-score head','#cb8b27')
for key,text in [('ba','Input a from SAME [QA]'),('bm','[BQ.M] image_to_text\nLinear256 -> 768, no bias + normalize'),('words','5 Quality phrases + class contexts\n[5,4,768], independent from [C]'),('bt','SAME frozen Transformer [T]\nGradients pass to class contexts'),('refs','5 normalized text references'),('logits','image-text similarity / temperature\n5 logits -> soft-label CE, lambda=.005'),('bnote','blend=0: no score mixing into Quality\nDoes NOT pass through U or text_projector')]:d.node(key,text,AMBER)
d.chain('ba','bm','logits');d.chain('words','bt','refs','logits');d.edge('logits','bnote',dashed=True);d.end()
d.node('note','SHARING: AG / AT / AP are independent adapters; AHG / AHP independent heads.\nPQ, PI, PT and DQ, DI, DT are PRETRAINING modules, not downstream SimSiam heads.\nQG/QP/NG/N/L/XI/XT/AHG/AHP/G exist downstream but are reset in representation scope.\nLegacy s_final computed using fusion_logit, not supervised; no independent Final regression head.\nOptional text_gate / extra attention / V-Mamba / distribution / Align bank / HA-HE are OFF.',GRAY)
d.save()
