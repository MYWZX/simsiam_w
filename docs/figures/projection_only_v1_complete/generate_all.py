#!/usr/bin/env python3
"""Build native editable paper figures; PDF/PNGs render only from the final deck."""
from pathlib import Path
from copy import deepcopy
import argparse,subprocess,shutil,json,hashlib,zipfile
import fitz
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
import drawing_primitives as d
import generate_srm as sr
HERE=Path(__file__).resolve().parent
# Reference palette: blue visual towers, cyan tensors, violet projectors,
# orange adapters/predictors, pale cyan Quality and pale peach Alignment.
d.C.update(ink='223047',muted='687386',line='4C5969',border='CDD9E7',vision='DCEBFF',feat='669CDD',proj='EDE2FF',text='F1E6FF',tb='AD88DB',q='EDFAFC',qb='96D9E0',qmod='D3F0F5',a='FFF3E8',ab='E8BA94',amod='FFE0C9',transfer='AC62D5',train='8492A2')
P=None

def text(sh,x,y,w,h,t,size=22,bold=False,c='ink',left=False,name='Text'):
 return d.txt(sh,x,y,w,h,t,size,bold,c,PP_ALIGN.LEFT if left else PP_ALIGN.CENTER,name)
def math(sh,x,y,w,h,t,size=25,name='Math'):return sr.math(sh,x,y,w,h,t,size,name)
def box(sh,x,y,w,h,title,fill='proj',size=23,sub=None,name=None):
 g=d.group(sh,name or title.replace('\n','_')).shapes
 d.shape(g,x,y,w,h,fill,'tb' if fill in ('proj','text') else ('ab' if fill=='amod' else 'border'),name=(name or title)+'_Body',width=1.25)
 if sub:
  text(g,x+.07,y+.09,w-.14,h*.45,title,size,True,name=(name or title)+'_Title')
  text(g,x+.07,y+h*.52,w-.14,h*.35,sub,size-5,c='muted',name=(name or title)+'_Sub')
 else:text(g,x+.06,y+.05,w-.12,h-.10,title,size,True,name=(name or title)+'_Label')
 return g

def mathbox(sh,x,y,w,h,symbol,sub,fill='proj'):
 g=d.group(sh,'Math_module_'+symbol).shapes
 d.shape(g,x,y,w,h,fill,'tb',name='Math_module_body_'+symbol,width=1.25)
 math(g,x+.06,y+.06,w-.12,h*.48,symbol,24,'Math_module_label_'+symbol)
 text(g,x+.06,y+h*.58,w-.12,h*.28,sub,17,c='muted')
 return g

def flow(sh,pts,name,kind='flow',arrow=True):return d.edge(sh,pts,kind,name,arrow)
def feat(sh,x,y,s,label=None,fill='vision'):
 g=d.group(sh,'Feature_'+s).shapes
 d.shape(g,x-.34,y-.28,.68,.56,fill,'feat',MSO_SHAPE.ROUNDED_RECTANGLE,'Tensor_'+s)
 math(g,x-.32,y-.27,.64,.54,s,25,'Symbol_'+s)
 if label:text(g,x-.85,y+.35,1.7,.31,label,16,c='muted',name='Dimension_'+s)
def op(sh,x,y,s):d.op(sh,x,y,s,.48,'white')
def panel(sh,x,y,w,h,title,fill='white',size=24):
 d.shape(sh,x,y,w,h,fill,'border',name=title+'_Region',width=1.1)
 text(sh,x+.20,y+.12,w-.40,.43,title,size,True,left=True,name=title+'_Heading')
def header(sh,title,sub):
 text(sh,.45,.23,19.1,.64,title,31,True,left=True,name='Figure_Title')
 text(sh,.47,.96,19.0,.39,sub,20,c='muted',left=True,name='Figure_Subtitle')
def slide(title,sub):
 s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white')
 G=d.group(s.shapes,title).shapes;header(G,title,sub);return s,G

def footer(sh,t):text(sh,.48,12.82,19.04,.31,t,17,c='muted',name='Figure_Notes')
def module(sh,x,y,w,h,name,caption,fill):return box(sh,x,y,w,h,name,fill,25,caption,name)
def icon(sh,x,y,w=1.3,h=.9):
 d.shape(sh,x,y,w,h,'DCEBFF','feat',MSO_SHAPE.RECTANGLE,'Input_image')
 d.shape(sh,x+.15,y+.12,.18,.18,'FFFFFF',None,MSO_SHAPE.OVAL,'Input_sun')
 d.shape(sh,x+.10,y+.42,.65,.35,'5CBBCB',None,MSO_SHAPE.ISOSCELES_TRIANGLE,'Input_mountain1')
 d.shape(sh,x+.55,y+.33,.60,.44,'8BC8ED',None,MSO_SHAPE.ISOSCELES_TRIANGLE,'Input_mountain2')
def tensor(sh,x,y,w=1.15,h=.58):
 for i in range(3):
  d.shape(sh,x+i*.07,y-i*.06,w,h,'A6ECF2' if i<2 else '58D2E1','24AEC6',MSO_SHAPE.RECTANGLE,'Local_tensor_'+str(i),.6)
 for k in range(1,6):flow(sh,[(x+.14+k*w/6,y-.12),(x+.14+k*w/6,y-.12+h)],'Tensor_grid_'+str(k),arrow=False)
def legend(sh,y):
 for x,kind,lab in [(.6,'flow','Forward data'),(4.0,'supervision','Training supervision'),(9.2,'transfer','Parameter initialization')]:
  flow(sh,[(x,y+.14),(x+.62,y+.14)],'Legend_'+kind,kind);text(sh,x+.73,y-.05,3.5,.36,lab,17,left=True)
 text(sh,14.1,y-.05,5.3,.36,'● feature fork     reuse = same tensor',17,c='muted')

def overview():
 s,G=slide('Projection-only multimodal assessment','Shared representations, separate Quality and Alignment modules; no cross-attention fusion.')
 # Stage panels.
 panel(G,.25,1.60,19.50,5.90,'(a) Quality-aware multimodal pretraining',size=25)
 icon(G,.65,2.58,1.40,.96);text(G,.50,3.67,1.70,.85,'Views I₁ / I₂\nClean Iᶜ',19,True)
 box(G,.52,5.09,1.67,.92,'Prompt P\nAnswer A','text',18)
 module(G,2.92,2.85,3.18,3.00,'SRE','Shared representation encoder\nCLIP · ResNet · Text','vision')
 flow(G,[(2.05,3.08),(2.92,3.08)],'Images_to_SRE')
 flow(G,[(2.19,5.55),(2.92,5.55)],'Texts_to_SRE')
 text(G,2.98,6.46,3.07,.45,'shared θ across views',18,c='muted')
 # Independent clean-view heads, with explicit inputs.
 module(G,7.28,4.07,3.08,.85,'QAM','Quality module','qmod')
 module(G,7.28,5.30,3.08,.85,'AAM','Alignment module','amod')
 flow(G,[(6.10,4.49),(7.28,4.49)],'Clean_quality_inputs');text(G,6.13,3.96,1.10,.37,'g, P₀, n',15)
 flow(G,[(6.10,5.73),(7.28,5.73)],'Clean_align_inputs');text(G,6.12,5.18,1.15,.35,'vᶜ, P₀, t',14)
 # Relation module calls. Signature text replaces ambiguous crossed wires.
 panel(G,12.03,2.38,7.26,3.89,'Four SRM calls · two shared predictors',fill='F8F5FF',size=20)
 calls=[('Image–Image','v₁, v₂','1'),('Image–Prompt','aᶜ, tP','a'),('Image–Answer','aᶜ, tA','c'),('Prompt–Answer','tP, tA','a·c')]
 for i,(name,pair,w) in enumerate(calls):
  yy=3.01+i*.75
  text(G,12.22,yy,2.65,.28,name,18,True,left=True)
  text(G,14.83,yy,1.50,.28,pair,18)
  box(G,16.62,yy-.06,1.10,.43,'SRM','proj',18,name='SRM_call_'+str(i))
  flow(G,[(16.33,yy+.15),(16.62,yy+.15)],'Pair_to_SRM_'+str(i))
  text(G,17.91,yy,1.12,.28,'w = '+w,17,c='muted')
 # Direct stem features for I-I and Answer; AAM taps for image-text.
 flow(G,[(6.10,3.10),(11.25,3.10),(11.25,3.43),(12.03,3.43)],'Direct_relation_features')
 text(G,6.40,2.62,4.43,.35,'v₁, v₂ from PI; tA from PT',18,c='muted')
 text(G,7.28,3.54,3.08,.31,'Clean-view assessment',18,True,c='muted')
 text(G,12.22,5.87,6.88,.28,'I–I: Quality task · text relations: Alignment task · PCGrad',15,c='muted')
 flow(G,[(10.36,5.73),(11.30,5.73),(11.30,5.43),(12.03,5.43)],'AAM_relation_features')
 text(G,10.38,6.00,1.59,.31,'aᶜ, tP',17,c='muted')
 box(G,7.28,6.65,11.99,.47,'Clean targets: Quality / Alignment · Components · Anchors + Ordinal · Local / word relations','q',17,name='Clean_supervision')
 flow(G,[(10.36,4.49),(10.78,4.49),(10.78,5.65)],'Quality_targets_upper','supervision',arrow=False)
 flow(G,[(10.78,5.81),(10.78,6.65)],'Quality_targets','supervision')
 flow(G,[(8.82,6.15),(8.82,6.65)],'Alignment_targets','supervision')
 # Cross-stage migration is a distinct purple dashed arrow, never a data wire.
 flow(G,[(3.28,7.85),(16.72,7.85)],'Pretrain_to_downstream','transfer')
 text(G,3.80,7.53,12.4,.28,'Representation initialization only · downstream scoring heads / matching layers / score gates reset',17,c='transfer')
 panel(G,.25,8.21,19.50,3.96,'(b) Downstream quality and alignment assessment',size=25)
 icon(G,.65,9.33,1.4,.96);box(G,.52,10.75,1.67,.61,'Prompt P','text',18)
 module(G,2.92,9.00,3.18,3.00,'SRE','Same architecture\nInitialized from pretraining','vision')
 flow(G,[(2.05,9.81),(2.92,9.81)],'Downstream_image')
 flow(G,[(2.19,11.05),(2.92,11.05)],'Downstream_text')
 module(G,8.04,9.13,3.08,.85,'QAM','Quality module','qmod')
 module(G,8.04,10.70,3.08,.85,'AAM','Alignment module','amod')
 flow(G,[(6.10,9.56),(8.04,9.56)],'Downstream_quality');text(G,6.22,9.03,1.70,.32,'g, P₀, n',18)
 flow(G,[(6.10,11.13),(8.04,11.13)],'Downstream_alignment');text(G,6.22,10.60,1.70,.32,'v, P₀, t',18)
 box(G,15.09,9.13,3.70,.85,'Perceptual quality','qmod',23,name='Quality_output')
 box(G,15.09,10.70,3.70,.85,'Text–image alignment','amod',22,name='Alignment_output')
 flow(G,[(11.12,9.56),(15.09,9.56)],'Quality_score');math(G,12.00,9.03,2.1,.39,'S_{Q}',24)
 flow(G,[(11.12,11.13),(15.09,11.13)],'Align_score');math(G,12.00,10.60,2.1,.39,'S_{A}',24)
 legend(G,12.41)
 footer(G,'projection_only_v1 · 5 trainable context tokens · frozen CLIP text weights · quality text-bank CE OFF · no negative push')
 s.notes_slide.notes_text_frame.text='Overview abstracts multiple exact feature inputs into labeled feature bundles. SRE shared weights within each stage; pretraining and downstream are separately initialized/trained, not tied weights. See module pages for exact paths and transfer specification in README. Four SRM weights are per-sample values, not external lambda coefficients. tA bypasses AAM. Clean supervised objectives remain; additional local/word SRM-family losses are outside four main calls. Trainable context tokens are not frozen.'


def sre():
 s,G=slide('SRE | Shared Representation Encoder','One image produces independent global and local features; text uses five trainable context tokens.')
 panel(G,.35,1.72,19.30,3.10,'Global visual path',fill='F3F7FF')
 icon(G,.70,2.85,1.20,.83)
 box(G,2.70,2.72,2.54,1.05,'CLIP ViT-L/14','vision',23,'trainable image tower')
 flow(G,[(1.90,3.26),(2.70,3.26)],'Image_to_CLIP')
 feat(G,6.10,3.26,'f','768D · L2')
 flow(G,[(5.24,3.26),(5.76,3.26)],'CLIP_to_f')
 box(G,7.23,2.72,2.52,1.05,'Global projection','proj',21,'Linear 768 → 256')
 flow(G,[(6.44,3.26),(7.23,3.26)],'f_to_global_projection')
 feat(G,10.65,3.26,'g','256D')
 flow(G,[(9.75,3.26),(10.31,3.26)],'Global_g')
 box(G,12.01,2.72,2.30,1.05,'PI','proj',25,'image projector + L2')
 flow(G,[(10.99,3.26),(12.01,3.26)],'g_to_PI')
 feat(G,15.20,3.26,'v','256D')
 flow(G,[(14.31,3.26),(14.86,3.26)],'PI_to_v')
 text(G,16.10,2.86,3.06,.80,'g → Quality\nv → Alignment / I–I',20,True)
 panel(G,.35,5.10,19.30,2.95,'Independent local path',fill='EFFBFC')
 icon(G,.70,6.18,1.20,.83)
 box(G,2.70,6.02,2.54,1.05,'ResNet-50','vision',23,'layer3 · trainable')
 flow(G,[(1.90,6.55),(2.70,6.55)],'Image_to_ResNet')
 tensor(G,5.85,6.32,.93,.55);text(G,5.34,7.14,1.95,.35,'14 × 14 × 1024',17,c='muted')
 flow(G,[(5.24,6.55),(5.85,6.55)],'ResNet_to_raw')
 box(G,7.23,6.02,2.52,1.05,'Local projection 1','proj',20,'1×1 Conv 1024 → 256')
 flow(G,[(6.92,6.55),(7.23,6.55)],'Raw_to_L1')
 box(G,10.45,6.02,2.52,1.05,'Local projection 2','proj',20,'1×1 Conv 256 → 256')
 flow(G,[(9.75,6.55),(10.45,6.55)],'L1_to_L2')
 tensor(G,13.65,6.32,.93,.55);math(G,14.80,6.20,1.0,.60,'P_{0}',26)
 flow(G,[(12.97,6.55),(13.65,6.55)],'Local_P0')
 text(G,16.10,6.05,3.06,.99,'196 × 256 tokens\nBypasses PI',21,True)
 panel(G,.35,8.34,19.30,3.25,'Text path',fill='FAF5FF')
 box(G,.70,9.68,1.80,.72,'Prompt / Answer','text',17)
 text(G,2.92,9.02,2.48,.32,'5 learnable tokens',18,True,c='transfer')
 for i in range(5):d.shape(G,3.04+i*.43,9.53,.31,.34,'D59AFA','tb',MSO_SHAPE.RECTANGLE,'Learnable_context_'+str(i))
 box(G,6.10,9.50,2.55,1.05,'CLIP text encoder','text',21,'frozen weights')
 d.lock(G,8.11,9.20)
 flow(G,[(2.50,10.04),(5.06,10.04)],'Text_tokens_to_concat')
 op(G,5.30,10.04,'C')
 flow(G,[(5.54,10.04),(6.10,10.04)],'Concatenated_tokens_to_encoder')
 flow(G,[(4.10,9.53),(4.10,9.39),(5.30,9.39),(5.30,9.80)],'Trainable_context_input')
 text(G,4.78,10.67,1.05,.27,'token concat.',13,c='muted')
 box(G,10.02,9.50,2.30,1.05,'Text projection T','proj',20,'Linear 768 → 256')
 flow(G,[(8.65,10.04),(10.02,10.04)],'Text_embedding_to_T');text(G,8.77,10.22,1.1,.30,'768D · L2',15,c='muted')
 box(G,13.40,9.50,2.00,1.05,'PT','proj',25,'text projector + L2')
 flow(G,[(12.32,10.04),(13.40,10.04)],'T_to_PT')
 feat(G,16.35,10.04,'t','256D',fill='text');flow(G,[(15.40,10.04),(16.01,10.04)],'PT_to_t')
 text(G,17.18,9.53,2.05,1.05,'Prompt → AAM\nAnswer → SRM',19,True)
 box(G,.70,11.92,2.15,.59,'Image I','vision',20)
 box(G,4.30,11.92,6.00,.59,'Lightweight NSS statistics','qmod',21)
 flow(G,[(2.85,12.215),(4.30,12.215)],'Image_to_NSS')
 feat(G,12.00,12.215,'n',fill='qmod')
 flow(G,[(10.30,12.215),(11.66,12.215)],'Raw_NSS_descriptor')
 text(G,13.00,12.02,5.5,.36,'6D per patch → QAM',20,True)
 footer(G,'PI/PT: separate 256 → 512 → 512 → 256 MLPs. Shared SRE weights across calls; g goes directly to Quality and P₀ bypasses PI.')
 s.notes_slide.notes_text_frame.text='Global semantic projection is backbone.fusion.semantic_proj Linear768->256; no cross-attention module instantiated. g and P0 remain independent. ResNet local_encoder.proj followed by local_patch_proj are separate 1x1 projections, kept intact. CLIP image semantic output and text encoder output are normalized before downstream linears. Frozen CLIP text tower remains differentiable wrt trainable five prefix context embeddings. PI/PT separate weights, shared only across calls of the same modality. Raw NSS n computed from image separately, input to QAM. See QAM.'


def qam():
 s,G=slide('QAM | Quality Assessment Module','Global residual scoring and NSS-guided local scoring remain separate until scalar score mixing.')
 panel(G,.35,1.72,19.30,3.00,'Global quality',fill='q')
 feat(G,1.10,3.38,'g','256D')
 box(G,2.65,2.96,2.15,.84,'QA','amod',25,'MLP + L2')
 flow(G,[(1.44,3.38),(2.65,3.38)],'g_to_QA');d.dot(G,1.95,3.38)
 feat(G,5.65,3.38,'q_{A}')
 flow(G,[(4.80,3.38),(5.31,3.38)],'QA_to_qA')
 box(G,6.76,2.96,2.66,.84,'Scale residual','proj',20,'α · sg(‖g‖)')
 flow(G,[(5.99,3.38),(6.76,3.38)],'qA_to_scale')
 op(G,10.15,3.38,'+')
 flow(G,[(9.42,3.38),(9.91,3.38)],'Residual_to_add')
 flow(G,[(1.95,3.38),(1.95,2.49),(10.15,2.49),(10.15,3.14)],'Global_identity_skip')
 box(G,11.45,2.96,2.15,.84,'Global head','vision',23,'MLP → sigmoid')
 flow(G,[(10.39,3.38),(11.45,3.38)],'Global_residual_to_score')
 feat(G,14.45,3.38,'s_{G}');flow(G,[(13.60,3.38),(14.11,3.38)],'Global_quality_scalar')
 # Local signal path.
 panel(G,.35,5.00,19.30,4.01,'Local quality',fill='q')
 feat(G,1.10,6.56,'P_{0}','196 × 256')
 op(G,3.05,6.56,'+')
 flow(G,[(1.44,6.56),(2.81,6.56)],'P0_to_add')
 box(G,4.13,6.14,2.08,.84,'QL','amod',25,'res. adapter + L2')
 flow(G,[(3.29,6.56),(4.13,6.56)],'NSS_token_to_QL')
 feat(G,7.05,6.56,'r')
 flow(G,[(6.21,6.56),(6.71,6.56)],'QL_tokens')
 box(G,9.05,6.14,2.31,.84,'Weighted pool','qmod',20,'sum over tokens + L2')
 flow(G,[(7.39,6.56),(9.05,6.56)],'Local_tokens_to_pool')
 feat(G,12.22,6.56,'q_{L}')
 flow(G,[(11.36,6.56),(11.88,6.56)],'Pooled_qL')
 box(G,13.30,6.14,2.15,.84,'Local head','vision',23,'MLP → sigmoid')
 flow(G,[(12.56,6.56),(13.30,6.56)],'qL_to_head')
 feat(G,16.30,6.56,'s_{L}');flow(G,[(15.45,6.56),(15.96,6.56)],'Local_quality_scalar')
 feat(G,1.10,8.00,'n','NSS · 6D',fill='qmod')
 box(G,2.20,7.63,2.04,.74,'NSS MLP','proj',20)
 flow(G,[(1.44,8.00),(2.20,8.00)],'NSS_to_embed');d.dot(G,1.70,8.00)
 flow(G,[(3.05,7.63),(3.05,6.80)],'Embedded_NSS_to_add')
 box(G,6.18,7.63,2.52,.74,'NSS gate','qmod',21,'MLP(r, n) → softmax')
 flow(G,[(1.70,8.00),(1.70,8.68),(5.61,8.68),(5.61,8.00),(6.18,8.00)],'Raw_NSS_to_gate')
 d.dot(G,7.70,6.56);flow(G,[(7.70,6.56),(7.70,7.63)],'r_to_gate')
 flow(G,[(8.70,8.00),(10.20,8.00),(10.20,6.98)],'Gate_weights_to_pool')
 text(G,10.36,7.55,1.40,.34,'weights w',16,c='muted')
 # Quality score mix, no cross-attention or feature fusion implied.
 box(G,17.12,4.52,2.01,.92,'Score mix','qmod',22,'(1−β)sG + βsL')
 flow(G,[(14.79,3.38),(18.13,3.38),(18.13,4.52)],'Global_to_mix')
 flow(G,[(16.64,6.56),(18.13,6.56),(18.13,5.44)],'Local_to_mix')
 math(G,17.25,7.50,1.75,.56,'S_{Q}',31)
 flow(G,[(19.13,4.98),(19.42,4.98),(19.42,7.78),(19.00,7.78)],'Quality_output')
 # Separate, explicit feature reuse inset, avoiding unreadable long crossings.
 panel(G,.35,9.43,19.30,2.66,'Pretraining only · reuse qA and qL above',fill='FAFCFD',size=21)
 feat(G,1.25,10.83,'q_{A}');feat(G,2.45,10.83,'q_{L}')
 text(G,.62,11.34,2.45,.31,'same features · reuse',15,c='muted')
 flow(G,[(2.79,10.83),(3.35,10.83)],'Quality_training_feature_bundle');d.dot(G,3.35,10.83)
 box(G,4.12,10.13,4.50,.72,'q = norm[(1−ρ)qA + ρqL]','qmod',20)
 box(G,10.03,10.13,8.62,.72,'Fixed component anchors + ordinal targets','q',22)
 flow(G,[(3.35,10.83),(3.35,10.49),(4.12,10.49)],'Feature_mix_inputs')
 flow(G,[(8.62,10.49),(10.03,10.49)],'Anchors_ordinal','supervision')
 box(G,4.12,11.18,4.50,.61,'C(qA, qL) = [qA, qL, |qA−qL|]','qmod',18)
 box(G,10.03,11.18,8.62,.61,'H_art / H_aes → component regression + ranking','q',20)
 flow(G,[(3.35,10.83),(3.35,11.48),(4.12,11.48)],'Auxiliary_feature_concat')
 flow(G,[(8.62,11.48),(10.03,11.48)],'Component_auxiliary','supervision')
 footer(G,'β and α are learned scalars. “+” adds features; score mix combines scalar predictions. Quality text-bank CE stays disabled.')
 s.notes_slide.notes_text_frame.text='QAM g directly bypasses PI. qA=normalize QA(g); hG=g+alpha*stopgrad(norm(g))*qA. H_QG hG scores after residual. QA is 256512512256. Local raw six-dimensional NSS embeds and adds to P0; r=normalize QL(P0+NSSMLP(n)); weights=softmax NSSgate(concat(r,n)); qL=normalize sum(weights*r); H_QL applied after pooling. qA/qL are not added directly to score. Beta scalar not sample gate. Training inset reuses existing qA/qL and produces q for anchors/ordinal plus concat for auxiliary heads; no extra networks or quality text CE. n is lightweight MSCN/gradient statistics, not full fitted NSS distribution.'


def aam():
 s,G=slide('AAM | Alignment Assessment Module','Global matching and text-guided patch evidence produce two scores, combined by an adaptive gate.')
 panel(G,.35,1.72,19.30,9.71,'Alignment computation',fill='a')
 # Global row.
 feat(G,1.10,3.13,'v','norm PI(g)')
 box(G,2.65,2.71,2.02,.84,'AG','amod',24,'residual adapter')
 flow(G,[(1.44,3.13),(2.65,3.13)],'v_to_AG')
 feat(G,5.55,3.13,'a_{G}');flow(G,[(4.67,3.13),(5.21,3.13)],'aG')
 op(G,7.15,3.13,'C');flow(G,[(5.89,3.13),(6.91,3.13)],'aG_to_concat')
 box(G,8.28,2.71,2.28,.84,'Global head','vision',24,'MLP → sigmoid')
 flow(G,[(7.39,3.13),(8.28,3.13)],'Global_pair_to_head')
 feat(G,11.42,3.13,'s_{G}');flow(G,[(10.56,3.13),(11.08,3.13)],'Global_align_score')
 text(G,8.14,3.92,3.13,.37,'Global image–text match',18,True,c='A56F48')
 # Patch row, TPM fully drawn on its own page.
 feat(G,1.10,6.02,'P_{0}','bypasses PI')
 box(G,2.65,5.60,2.02,.84,'AP','amod',24,'residual adapter')
 flow(G,[(1.44,6.02),(2.65,6.02)],'P0_to_AP')
 feat(G,5.55,6.02,'U');flow(G,[(4.67,6.02),(5.21,6.02)],'AP_patch_tokens')
 module(G,7.00,5.53,2.82,.98,'TPM','Text-guided patch match','qmod')
 flow(G,[(5.89,6.02),(7.00,6.02)],'U_to_TPM')
 feat(G,10.68,6.02,'e');flow(G,[(9.82,6.02),(10.34,6.02)],'Pooled_evidence')
 op(G,12.20,6.02,'C');flow(G,[(11.02,6.02),(11.96,6.02)],'Evidence_to_pair')
 box(G,13.30,5.60,2.10,.84,'Local head','vision',23,'MLP → sigmoid')
 flow(G,[(12.44,6.02),(13.30,6.02)],'Local_pair_to_head')
 feat(G,16.17,6.02,'s_{P}');flow(G,[(15.40,6.02),(15.83,6.02)],'Local_align_score')
 # Text features fan out at explicit fork points, with a gap at the unrelated patch wire.
 feat(G,1.10,8.97,'t','norm PT(T(Eₜ(P)))',fill='text')
 box(G,2.65,8.55,2.02,.84,'AT','amod',24,'residual adapter')
 flow(G,[(1.44,8.97),(2.65,8.97)],'t_to_AT')
 feat(G,5.55,8.97,'a_{T}',fill='text');flow(G,[(4.67,8.97),(5.21,8.97)],'aT')
 flow(G,[(5.89,8.97),(6.35,8.97),(6.35,6.10)],'Text_to_global_pair_lower',arrow=False)
 flow(G,[(6.35,5.94),(6.35,4.48),(7.15,4.48),(7.15,3.37)],'Text_to_global_pair_upper')
 # At x7.15 the feature route is deliberately interrupted across TPM; route outside instead.
 # This text-to-global line should pass left of TPM to avoid crossing that module.
 # Independent fork routes preserve feature identity.
 d.dot(G,6.35,8.97)
 flow(G,[(6.35,8.97),(8.41,8.97),(8.41,6.51)],'Text_to_TPM')
 flow(G,[(8.41,8.97),(12.20,8.97),(12.20,6.26)],'Text_to_local_pair');d.dot(G,8.41,8.97)
 text(G,8.57,7.48,2.90,.37,'same aT feature',17,c='muted')
 # The gate consumes the actual four features/scores, shown explicitly as a reused input bundle.
 box(G,13.00,8.57,3.25,.80,'Adaptive gate','amod',22,'[aG, aT, sG, sP] → γ')
 text(G,13.02,9.56,3.21,.35,'reuses features + scores',16,c='muted')
 box(G,17.12,4.15,2.01,.95,'Score mix','amod',22,'γsG + (1−γ)sP')
 flow(G,[(11.76,3.13),(18.13,3.13),(18.13,4.15)],'Global_align_to_mix')
 flow(G,[(16.51,6.02),(18.13,6.02),(18.13,5.10)],'Local_align_to_mix')
 flow(G,[(16.25,8.97),(17.10,8.97),(17.10,6.10)],'Adaptive_gamma_lower',arrow=False)
 flow(G,[(17.10,5.94),(17.10,5.48),(17.47,5.48),(17.47,5.10)],'Adaptive_gamma_to_mix')
 text(G,17.16,7.56,1.53,.32,'γ per image',16,c='muted')
 math(G,17.40,10.11,1.46,.54,'S_{A}',32)
 flow(G,[(19.13,4.625),(19.42,4.625),(19.42,10.38),(18.86,10.38)],'Alignment_output')
 text(G,2.12,10.32,10.40,.44,'C(x, t) = [x, t, |x−t|]     U = AP(P₀)',23,c='muted')
 # Pretraining exports with exact definitions; Answer bypass is explicit.
 panel(G,.35,11.76,19.30,.76,'Pretraining taps: aᶜ = norm(aG); tP = norm(W_T(aT)). Answer tA comes directly from SRE, bypassing AAM.',size=18)
 footer(G,'AG / AP / AT are separate adapters. The adaptive gate reuses the four labeled inputs; TPM performs dot-product pooling, not cross-attention.')
 s.notes_slide.notes_text_frame.text='AAM input v and t already PI/PT projected. AG/AP/AT separate residual adapters. C concatenates x,t,abs difference. H_AG and H_AP output sigmoid scalars. Gate gamma= sigmoid MLP(concat(aG,aT,sG,sP)); output gamma*sG+(1-gamma)*sP. TPM details next page. Main Prompt training relation tP is norm WT(aT), not raw PT output; aC=norm aG. Answer tA bypasses AT/WT. Reused gate inputs named explicitly rather than long cluttered connecting wires.'


def tpm():
 s,G=slide('TPM | Text-guided Patch Matching','Use text to weight patches; pool the original adapted patch tokens to obtain local evidence.')
 panel(G,.35,1.72,19.30,8.90,'Matching → patch weights → local evidence',fill='a')
 feat(G,1.13,3.72,'U','196 × 256')
 mathbox(G,3.03,3.30,2.32,.84,'W_{I}','Linear + L2')
 flow(G,[(1.47,3.72),(3.03,3.72)],'U_to_WI');d.dot(G,2.05,3.72)
 tensor(G,6.05,3.49,1.10,.55);flow(G,[(5.35,3.72),(6.05,3.72)],'Projected_patch_tokens')
 text(G,5.68,4.38,2.02,.39,'normalized patches',17,c='muted')
 feat(G,1.13,7.58,'a_{T}','256D',fill='text')
 mathbox(G,3.03,7.16,2.32,.84,'W_{T}','Linear + L2')
 flow(G,[(1.47,7.58),(3.03,7.58)],'aT_to_WT')
 feat(G,6.51,7.58,'t_{P}',fill='text');flow(G,[(5.35,7.58),(6.17,7.58)],'Normalized_text_match')
 box(G,8.77,5.20,2.37,1.02,'Dot product','qmod',22,'one similarity / patch')
 flow(G,[(7.29,3.72),(9.95,3.72),(9.95,5.20)],'Patch_to_dot')
 flow(G,[(6.85,7.58),(9.95,7.58),(9.95,6.22)],'Text_to_dot')
 box(G,12.12,5.20,2.13,1.02,'Softmax','proj',23,'over 196 patches')
 flow(G,[(11.14,5.71),(12.12,5.71)],'Similarity_to_softmax')
 # Editable illustrative patch-weight heatmap.
 vals=[['D1F3F6','A1E4ED','D1F3F6','E4F8FA'],['99DEE8','36BDD2','79D1E0','D1F3F6'],['D1F3F6','67CCDE','A1E4ED','E4F8FA']]
 for i,row in enumerate(vals):
  for j,c in enumerate(row):d.shape(G,15.12+j*.22,5.32+i*.22,.20,.20,c,None,MSO_SHAPE.RECTANGLE,'Illustrative_weight_'+str(i)+'_'+str(j))
 flow(G,[(14.25,5.71),(15.05,5.71)],'Softmax_weights')
 text(G,14.70,6.31,1.90,.64,'weights w\nillustrative',16,c='muted')
 box(G,17.03,5.20,2.12,1.02,'Weighted sum','qmod',22,'e = Σᵢ wᵢ Uᵢ')
 flow(G,[(16.03,5.71),(17.03,5.71)],'Weights_to_sum')
 # Pool original U; not the matching projection W_I(U).
 flow(G,[(2.05,3.72),(2.05,2.63),(18.09,2.63),(18.09,5.20)],'Unprojected_U_to_weighted_sum')
 text(G,9.00,2.10,6.3,.33,'Original U tokens · reused for pooling',20,True,c='muted')
 feat(G,18.09,8.70,'e','256D')
 flow(G,[(18.09,6.22),(18.09,8.42)],'Evidence_vector')
 box(G,3.02,9.02,8.82,.81,'tP is reused by SRM during pretraining','text',22)
 flow(G,[(6.51,7.86),(6.51,9.02)],'Prompt_relation_tap')
 text(G,.65,11.10,18.70,.85,'The matching projections produce attention weights only. The weighted values are U = AP(P₀).\nThis is scalar dot-product patch pooling, with no multi-head cross-attention or global–local fusion output.',21,c='muted')
 footer(G,'W_I and W_T are learned Linear matching layers; both reset for downstream training. Heatmap cells are an illustrative vector schematic.')
 s.notes_slide.notes_text_frame.text='p_img=normalize WI(U), p_txt=normalize WT(aT), weights=softmax(sum(p_img*p_txt,dimchannels),dimpatch), evidence=sum(weights*U). No scaling factor sqrtD in code. Evidence not additionally normalized before pair concat. Prompt tP aliases p_txt. WI/WT are distinct matching layers reset downstream. Figure heatmap illustrative, not an actual sample or measured result.'


def srm(out):
 # Keep the accepted simple directional structure; recolor to the reference palette.
 tmp=out/'_srm_source';tmp.mkdir(exist_ok=True)
 q=sr.build(tmp)
 src=Presentation(q).slides[0];s=P.slides.add_slide(P.slide_layouts[6])
 wrapper=s.shapes.add_group_shape();wrapper.name='SRM_native_editable_group'
 for e in src.shapes:
  if e.shape_type==13:raise AssertionError('No raster SRM module')
  wrapper.shapes._spTree.insert_element_before(deepcopy(e._element),'p:extLst')
 # Native child geometry is preserved under a simple uniform group transform.
 wrapper.left=Inches(0);wrapper.top=Inches(.72);wrapper.width=Inches(20);wrapper.height=Inches(11.75)
 # The source group bbox differs slightly from full slide; apply exact slide-sized child coordinates.
 xf=wrapper._element.grpSpPr.xfrm;xf.chOff.x=0;xf.chOff.y=0;xf.chExt.cx=Inches(16);xf.chExt.cy=Inches(9.4)
 s.notes_slide.notes_text_frame.text=src.notes_slide.notes_text_frame.text
 q.unlink();tmp.rmdir()

NAMES=['01_Framework','02_SRE','03_QAM','04_AAM','05_TPM','06_SRM']
def render(out):
 deck=out/'ProjectionOnly_Modular_Framework.pptx'
 from types import SimpleNamespace
 from pptx.oxml.ns import qn
 for s in P.slides:
  sr.attach(SimpleNamespace(slides=[s]))
  # Imported native SRM groups need unique IDs within the destination slide.
  ids=list(s.shapes._spTree.iter(qn('p:cNvPr')));mapping={}
  for i,node in enumerate(ids,1):
   mapping[node.get('id')]=str(i);node.set('id',str(i))
  for tag in ('a:stCxn','a:endCxn'):
   for node in s.shapes._spTree.iter(qn(tag)):
    node.set('id',mapping[node.get('id')])
  assert len({node.get('id') for node in ids})==len(ids)
 P.save(deck)
 # Native connectors get attachment references where endpoints exactly meet module midpoints.
 exe=shutil.which('soffice') or shutil.which('libreoffice')
 subprocess.run([exe,'--headless','--convert-to','pdf','--outdir',str(out),str(deck)],check=True,timeout=180)
 pdf=fitz.open(out/'ProjectionOnly_Modular_Framework.pdf');report=[]
 for i,page in enumerate(pdf):
  pix=page.get_pixmap(matrix=fitz.Matrix(6000/page.rect.width,6000/page.rect.width),alpha=False);pix.save(out/(NAMES[i]+'.png'))
  page.get_pixmap(matrix=fitz.Matrix(1800/page.rect.width,1800/page.rect.width),alpha=False).save(out/(NAMES[i]+'_preview.png'))
  assert not page.get_images(),(i,'Unexpected PDF raster object')
  assert '\ufffd' not in page.get_text(),(i,'Replacement glyph')
  spans=[v for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for v in l['spans']]
  outliers=[v['text'] for v in spans if not page.rect.contains(fitz.Rect(v['bbox']))]
  assert not outliers,(i,outliers)
  counts=dict(shapes=0,connectors=0,text_objects=0,pictures=0)
  def walk(shapes):
   for o in shapes:
    counts['shapes']+=1
    if o.shape_type==6:walk(o.shapes)
    if o.shape_type==9:counts['connectors']+=1
    if o.shape_type==13:counts['pictures']+=1
    if o.has_text_frame and o.text:counts['text_objects']+=1
  walk(P.slides[i].shapes)
  report.append(dict(name=NAMES[i],pixels=[pix.width,pix.height],editable_objects=counts,pdf_raster_images=0,out_of_page_text=outliers))
 (out/'render_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 P=Presentation();P.slide_width=Inches(20);P.slide_height=Inches(13.5)
 overview();sre();qam();aam();tpm();srm(out);render(out)
