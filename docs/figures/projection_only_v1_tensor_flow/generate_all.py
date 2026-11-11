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
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
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

def prism(sh,x,y,w,h,depth=.12,name='Tensor'):
 g=d.group(sh,name).shapes
 # Native editable faces; depth is purely illustrative, not a literal size axis.
 from pptx.dml.color import RGBColor
 def face(points,color,suffix):
  b=g.build_freeform(points[0][0]*1000,points[0][1]*1000,scale=Inches(1)/1000)
  b.add_line_segments([(xx*1000,yy*1000) for xx,yy in points[1:]],close=True);o=b.convert_to_shape();o.name=name+suffix
  for ref in o._element.findall('.//'+qn('a:effectRef')):ref.set('idx','0')
  o._element.spPr.append(OxmlElement('a:effectLst'))
  o.fill.solid();o.fill.fore_color.rgb=d.rgb(color);o.line.color.rgb=d.rgb('4269A0');o.line.width=Pt(.7)
 face([(x,y),(x+depth,y-depth),(x+w+depth,y-depth),(x+w,y)],'7EA4E5','_Top')
 face([(x+w,y),(x+w+depth,y-depth),(x+w+depth,y+h-depth),(x+w,y+h)],'345EAD','_Side')
 d.shape(g,x,y,w,h,'517FCB','4269A0',MSO_SHAPE.RECTANGLE,name+'_Front',.7)

def encoder(sh,x,y,w,h,title,subtitle,fill='8FADE1'):
 g=d.group(sh,title+'_Encoder').shapes
 b=g.build_freeform(x*1000,y*1000,scale=Inches(1)/1000);b.add_line_segments([((x+w)*1000,(y+.18)*1000),((x+w)*1000,(y+h-.18)*1000),(x*1000,(y+h)*1000)],close=True)
 o=b.convert_to_shape();o.name=title+'_Wedge'
 for ref in o._element.findall('.//'+qn('a:effectRef')):ref.set('idx','0')
 o._element.spPr.append(OxmlElement('a:effectLst')); o.fill.solid();o.fill.fore_color.rgb=d.rgb(fill);o.line.fill.background()
 text(g,x+.06,y+.24,w-.12,h*.40,title,22,True,c='white')
 text(g,x+.06,y+h*.61,w-.12,h*.20,subtitle,15,c='white')

def matrix(sh,x,y,w,h,name='P0_matrix'):
 d.shape(sh,x,y,w,h,'E2EEFF','669CDD',MSO_SHAPE.RECTANGLE,name)
 for k in range(1,6):flow(sh,[(x,y+k*h/6),(x+w,y+k*h/6)],name+'_row'+str(k),arrow=False)
 for k in range(1,5):flow(sh,[(x+k*w/5,y),(x+k*w/5,y+h)],name+'_col'+str(k),arrow=False)

def column(sh,x,y,w,h,title,size=23):
 from pptx.enum.dml import MSO_LINE_DASH_STYLE
 o=d.shape(sh,x,y,w,h,'white','border',name=title+'_Column',width=1.0)
 o.adjustments[0]=.025;o.line.dash_style=MSO_LINE_DASH_STYLE.DASH
 text(sh,x+.20,y+.12,w-.40,.43,title,size,True,left=True)

def overview():
 s,G=slide('From image tensors to Quality and Alignment scores','projection_only_v1 · dimensions shown for one image · tensor drawings are not to scale')
 # Simple vertical columns echo the supplied reference while keeping every input path explicit.
 column(G,.30,1.70,2.10,9.13,'Inputs',size=23)
 column(G,2.65,1.70,2.75,9.13,'Encoders',size=23)
 column(G,5.63,1.70,6.69,9.13,'Feature operations and tensors',size=22)
 icon(G,.68,2.83,1.32,.90);text(G,.45,3.88,1.80,.67,'Image I\n224 × 224 × 3',17,True)
 encoder(G,2.82,2.53,2.40,1.46,'CLIP image','ViT-L/14')
 encoder(G,2.82,5.30,2.40,1.46,'ResNet-50','layer3')
 encoder(G,2.82,9.15,2.40,1.46,'CLIP text','frozen weights','B398D6')
 d.lock(G,4.85,8.91)
 flow(G,[(2.00,3.28),(2.82,3.28)],'Image_to_visual');d.dot(G,2.27,3.28)
 flow(G,[(2.27,3.28),(2.27,6.03),(2.82,6.03)],'Same_image_to_local');d.dot(G,2.27,6.03)
 box(G,.52,8.54,1.66,.64,'Prompt P','text',18)
 box(G,.52,9.62,1.66,.72,'Answer A','text',18,'pretraining only')
 flow(G,[(2.18,8.86),(2.46,8.86),(2.46,9.68),(2.82,9.68)],'Prompt_to_text')
 flow(G,[(2.18,9.98),(2.82,9.98)],'Answer_to_shared_text')
 text(G,2.84,7.98,2.35,.38,'5 trainable ctx',18,True,c='transfer')
 for i in range(5):d.shape(G,2.97+i*.42,8.53,.28,.32,'D59AFA','tb',MSO_SHAPE.RECTANGLE,'Context_'+str(i))
 flow(G,[(4.08,8.85),(4.08,9.15)],'Context_to_text')
 # Global: CLIP returns a vector; L2 is an operation, Linear is a learned module.
 prism(G,5.86,3.12,.56,.28,name='CLIP_768_vector');flow(G,[(5.22,3.28),(5.86,3.28)],'Visual_embedding')
 text(G,5.54,3.70,1.20,.52,'768-D\nvector',16,c='muted')
 box(G,6.95,2.94,.72,.68,'L2','white',18)
 flow(G,[(6.54,3.28),(6.95,3.28)],'Vector_to_L2')
 box(G,8.03,2.86,2.32,.84,'Global Linear','proj',20,'768 → 256')
 flow(G,[(7.67,3.28),(8.03,3.28)],'L2_to_global_projection')
 feat(G,11.39,3.28,'g','256-D');flow(G,[(10.35,3.28),(11.05,3.28)],'Global_g')
 # Local: both convolutions are distinct learned channel projections.
 prism(G,5.79,5.77,.63,.62,.13,'ResNet_feature_map');flow(G,[(5.22,6.03),(5.79,6.03)],'Local_raw_map')
 text(G,5.56,6.51,1.37,.52,'14 × 14\n× 1024',16,c='muted')
 box(G,7.12,5.61,1.15,.84,'1×1 Conv','proj',16,'1024 → 256')
 box(G,8.67,5.61,1.15,.84,'1×1 Conv','proj',16,'256 → 256')
 flow(G,[(6.55,6.03),(7.12,6.03)],'Map_to_projection1')
 flow(G,[(8.27,6.03),(8.67,6.03)],'Projection1_to_projection2')
 matrix(G,10.91,5.72,.72,.62,'P0_token_matrix');flow(G,[(9.82,6.03),(10.91,6.03)],'Flatten_to_tokens')
 text(G,9.91,6.57,1.03,.34,'flatten',15,c='muted')
 math(G,10.70,6.44,1.16,.42,'P_{0}',25)
 text(G,10.30,6.89,1.60,.44,'196 × 256\ntoken matrix',15,c='muted')
 # Raw NSS is computed from the image; it is not a CLIP token.
 flow(G,[(2.27,6.03),(2.27,7.46),(4.08,7.46)],'Image_to_raw_NSS')
 box(G,4.08,7.09,2.34,.74,'NSS statistics','qmod',18)
 feat(G,8.05,7.46,'n','196 × 6',fill='qmod');flow(G,[(6.42,7.46),(7.71,7.46)],'NSS_n')
 # Text outputs are separate calls with shared weights. Only Prompt feeds downstream AAM.
 prism(G,5.86,9.83,.56,.28,name='Text_768_vector');flow(G,[(5.22,9.98),(5.86,9.98)],'Text_embedding')
 text(G,5.54,10.34,1.20,.31,'768-D',16,c='muted')
 box(G,6.95,9.64,.72,.68,'L2','white',18)
 flow(G,[(6.54,9.98),(6.95,9.98)],'Text_to_L2')
 box(G,8.03,9.56,1.29,.84,'Linear T','proj',18,'768 → 256')
 box(G,9.78,9.56,.98,.84,'PT + L2','proj',16,'256 → 256')
 flow(G,[(7.67,9.98),(8.03,9.98)],'Text_L2_to_T')
 flow(G,[(9.32,9.98),(9.78,9.98)],'T_to_PT')
 feat(G,11.39,9.98,'t','256-D',fill='text');flow(G,[(10.76,9.98),(11.05,9.98)],'Projected_text')
 # Task modules. Each feature has one real fork; unrelated wire crossings have gaps.
 module(G,15.06,2.68,2.91,2.62,'QAM','Quality assessment','qmod')
 flow(G,[(11.73,3.28),(15.06,3.28)],'g_to_QAM');d.dot(G,12.63,3.28)
 flow(G,[(11.63,6.03),(14.26,6.03),(14.26,4.76),(15.06,4.76)],'P0_to_QAM');d.dot(G,12.03,6.03)
 flow(G,[(8.39,7.46),(11.93,7.46)],'NSS_right1a',arrow=False)
 flow(G,[(12.13,7.46),(12.53,7.46)],'NSS_right1b',arrow=False)
 flow(G,[(12.73,7.46),(13.78,7.46),(13.78,6.13)],'NSS_right2',arrow=False)
 flow(G,[(13.78,5.93),(13.78,4.05),(15.06,4.05)],'NSS_to_QAM')
 module(G,15.06,7.77,2.91,3.05,'AAM','Alignment assessment','amod')
 # g -> PI belongs to alignment, while quality bypasses PI.
 flow(G,[(12.63,3.28),(12.63,5.93)],'g_to_PI_upper',arrow=False)
 flow(G,[(12.63,6.13),(12.63,8.42),(13.13,8.42)],'g_to_PI_lower')
 box(G,13.13,8.03,1.43,.78,'PI + L2','proj',20,'256 → 256')
 flow(G,[(14.56,8.42),(15.06,8.42)],'v_to_AAM')
 flow(G,[(12.03,6.03),(12.03,9.18),(15.06,9.18)],'P0_to_AAM')
 flow(G,[(11.73,9.98),(15.06,9.98)],'Prompt_t_to_AAM')
 text(G,12.59,10.29,2.31,.31,'Prompt only',16,c='muted')
 for y,sym,c in [(4.01,'S_{Q}','qmod'),(9.30,'S_{A}','amod')]:
  flow(G,[(17.97,y),(18.57,y)],'Score_'+sym)
  math(G,18.61,y-.36,1.03,.62,sym,30)
  text(G,18.44,y+.42,1.35,.34,'1 scalar',17,c='muted')
 # Compact training scope keeps inference tensors readable, full SRM expanded on page 6.
 panel(G,.30,11.28,19.40,1.09,'Pretraining only',fill='FAF8FF',size=20)
 text(G,3.31,11.46,15.97,.50,'Two SafeAug views → I–I; Clean view → labels / component anchors / ordinal; Prompt + Answer → SRM relations',18,c='muted')
 footer(G,'Baseline visual tower: trainable. Frozen-CLIP ablation: same shapes and paths. No cross-attention and no fused z. L2 has no learned weights.')
 s.notes_slide.notes_text_frame.text='Input image HWC2242243, tensor NCHW B3224224. CLIP-L14 visual257tokens width1024, outputs768D via CLS projection, normalized then semantic_proj768256 g. P0 from ResNet layer3 14x14x1024 -> local_encoder.proj1024256 -> local_patch_proj256256 -> flatten196256. CLIP patch256 tokens are not P0. Text tower shown once with shared calls P/A and learnable5 context. AAM only receives Prompt output; Answer output goes only to pretrain relations. NSS descriptor1966 from image. QAM actual outputs scalar score mixing and AAM actual outputs independent scalar. See detailed module slides and page7 for L2 and 1x1 conv semantics. Three-dimensional feature icons are schematic cuboids, never a claim that196x256 is a cube.'


def operations():
 s,G=slide('How to read the tensor operations','L2 is a fixed arithmetic operation; 1×1 projections are learned layers; flatten only changes the layout.')
 panel(G,.35,1.72,19.30,2.92,'1  L2 normalization · no learned parameters',fill='F3F7FF')
 box(G,.90,2.87,3.60,.75,'x = [3, 4]','vision',25)
 flow(G,[(4.50,3.245),(6.04,3.245)],'Example_to_L2')
 box(G,6.04,2.87,5.43,.75,'x / max(‖x‖₂, ε)','white',26)
 flow(G,[(11.47,3.245),(13.03,3.245)],'L2_example_result')
 box(G,13.03,2.87,5.46,.75,'[0.6, 0.8]','vision',27)
 text(G,.90,3.93,17.59,.38,'Same shape; divide each feature vector by its L2 norm, with ε protection. No learned weights; not computed across images.',19,c='muted')
 panel(G,.35,5.00,19.30,3.01,'2  Two independent 1×1 convolutions · learned channel projections',fill='FAF5FF')
 box(G,.90,6.14,3.60,.84,'14 × 14 × 1024','vision',25)
 box(G,5.46,6.14,3.80,.84,'Conv 1: 1024 → 256','proj',22)
 box(G,10.22,6.14,3.80,.84,'Conv 2: 256 → 256','proj',22)
 box(G,14.98,6.14,3.51,.84,'14 × 14 × 256','vision',25)
 for a,b in [(4.50,5.46),(9.26,10.22),(14.02,14.98)]:flow(G,[(a,6.56),(b,6.56)],'Channel_projection_'+str(a))
 text(G,.90,7.35,17.59,.35,'Each location mixes its channel values with a learned matrix. Spatial size stays 14×14; the two layers have different weights.',19,c='muted')
 panel(G,.35,8.36,19.30,3.77,'3  Feature map → token matrix · reshape only',fill='q')
 prism(G,1.30,9.66,1.57,1.22,.30,'Cuboid_feature_map')
 text(G,.90,11.12,3.60,.39,'14 × 14 × 256',24,True)
 flow(G,[(3.53,10.27),(7.01,10.27)],'Flatten_spatial_positions')
 text(G,3.90,9.58,2.80,.38,'flatten H × W',20,c='muted')
 matrix(G,7.01,9.46,2.21,1.62,'196_by_256_tokens')
 text(G,6.36,11.40,3.51,.37,'P₀: 196 × 256',24,True)
 text(G,10.41,9.49,8.29,1.57,'196 rows = 14×14 spatial positions\n256 columns = channels / features per position\nTotal: 196×256 = 50,176 values',23,left=True)
 footer(G,'A cuboid icon is an illustration of an H×W×C feature map. P₀ itself is a matrix (batch form B×196×256), not a physical cube.')
 s.notes_slide.notes_text_frame.text='L2 normalize last feature axis with epsilon floor, no parameters, differentiable. Conv1 1x1 1024->256, Conv2 1x1 256->256; same learned matrix at every location within a layer, different matrices between the two layers, no direct spatial aggregation in1x1. Example maps14x14x256 reshape196x256 preserves exactly50176 values, no pooling or averaging. PyTorch native map BCHW -> flatten(2).transpose(1,2) BNC.'

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
 text(G,.90,4.13,18.20,.39,'Inside CLIP: 16×16 patches + CLS → 257×1024 tokens → CLS (1024-D) → CLIP output projection (768-D)',19,c='muted')
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

NAMES=['01_Framework','02_SRE','03_QAM','04_AAM','05_TPM','06_SRM','07_Operations']
def render(out):
 deck=out/'Tensor_Flow_Framework.pptx'
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
 pdf=fitz.open(out/'Tensor_Flow_Framework.pdf');report=[]
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
 overview();sre();qam();aam();tpm();srm(out);operations();render(out)
