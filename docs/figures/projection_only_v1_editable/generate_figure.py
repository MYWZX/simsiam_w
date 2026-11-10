#!/usr/bin/env python3
"""Native editable PowerPoint is the only drawing source; PDF and PNG render from it.
Requires python-pptx, PyMuPDF, Pillow, and a LibreOffice/soffice executable.
Run: python generate_figure.py --output-dir ./rendered
"""
from pathlib import Path
import argparse, json, subprocess, shutil, hashlib, zipfile
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from PIL import ImageFont
import fitz

W,H=26,14.65
FONT='DejaVu Sans'
C={'ink':'253646','muted':'61717D','line':'435664','border':'CED8DD',
   'vision':'DEECF5','proj':'EDF4F9','q':'EAF7F5','qb':'9BC8C1','qmod':'D1ECE6',
   'a':'FFF2E6','ab':'DDB995','amod':'F8DEC5','text':'EEE8F7','tb':'BCAED4',
   'train':'82909B','transfer':'8965AA','white':'FFFFFF','feat':'4E8DAB'}
P=None; S=None; names=[]; nodes={}; edges=[]; fitted=[]; anchor_shapes=[]; connector_records=[]

def rgb(v):return RGBColor.from_string(C.get(v,v))
def nm(o,n):o.name=n;names.append(n);return o

def shape(sh,x,y,w,h,fill='white',stroke='line',kind=MSO_SHAPE.ROUNDED_RECTANGLE,name='shape',width=1.1):
 o=nm(sh.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h)),name)
 o.fill.solid();o.fill.fore_color.rgb=rgb(fill)
 for ref in o._element.findall('.//'+qn('a:effectRef')):ref.set('idx','0')
 o._element.spPr.append(OxmlElement('a:effectLst'))
 if stroke is None:o.line.fill.background()
 else:o.line.color.rgb=rgb(stroke);o.line.width=Pt(width)
 if .30<w<7 and .30<h<4:anchor_shapes.append(o)
 if kind==MSO_SHAPE.ROUNDED_RECTANGLE:
  try:o.adjustments[0]=.025 if w>7 else .10
  except (IndexError,ValueError):pass
 return o

def txt(sh,x,y,w,h,text,size=24,bold=False,color='ink',align=PP_ALIGN.CENTER,name='text',italic=False):
 o=nm(sh.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)),name)
 tf=o.text_frame;tf.clear();tf.word_wrap=False
 # Fit explicit lines without introducing unexpected word wrapping.
 fontname='DejaVuSans'+('-Bold' if bold else '')+'.ttf'
 bundled=Path(__file__).resolve().parent/'fonts'/fontname
 fontpath=str(bundled if bundled.exists() else Path('/usr/share/fonts/truetype/dejavu')/fontname)
 lines=text.split('\n'); original_size=size
 for _ in range(80):
  font=ImageFont.truetype(fontpath,round(size*8))
  maxwidth=max(font.getlength(line)/8 for line in lines)
  if maxwidth <= (w-.065)*72 and len(lines)*size*1.05 <= (h-.015)*72:break
  size-=.25
 if size<original_size:fitted.append(dict(name=name,original=original_size,actual=size,text=text))
 tf.margin_left=tf.margin_right=Inches(.025);tf.margin_top=tf.margin_bottom=0
 tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 for i,line in enumerate(text.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
  p.alignment=align;p.space_before=Pt(0);p.space_after=Pt(0)
  r=p.add_run();r.text=line;r.font.name=FONT;r.font.size=Pt(size);r.font.bold=bold;r.font.italic=italic;r.font.color.rgb=rgb(color)
 return o

def group(sh,name):return nm(sh.add_group_shape(),name)
def block(sh,x,y,w,h,title,id=None,fill='proj',stroke='line',size=23,head=False):
 g=group(sh,'Module_'+(id or title.replace('\n','_')))
 kind=MSO_SHAPE.RECTANGLE if id in ('G','L','T','PI','PT') else MSO_SHAPE.ROUNDED_RECTANGLE
 b=shape(g.shapes,x,y,w,h,fill,stroke,kind,'Body_'+(id or title),1.7 if head else 1.1)
 txt(g.shapes,x+.035,y+.025,w-.07,h-.05,title,size,head,name='Label_'+(id or title))
 if id:nodes.setdefault(id,[]).append((x,y,w,h,b))
 return (x,y,w,h,b)

def dot(sh,x,y,r=.035,color='line',name='feature_fork'):
 return shape(sh,x-r,y-r,r*2,r*2,color,None,MSO_SHAPE.OVAL,name)

def edge(sh,pts,kind='flow',name='edge',arrow=True):
 g=group(sh,'Connector_'+name)
 col={'flow':'line','supervision':'train','transfer':'transfer','text':'tb'}[kind]
 for i,((x1,y1),(x2,y2)) in enumerate(zip(pts,pts[1:])):
  if abs(x2-x1)+abs(y2-y1)<1e-7:continue
  o=nm(g.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1), Inches(x2), Inches(y2)),name+f'_{i}')
  for ref in o._element.findall('.//'+qn('a:effectRef')):ref.set('idx','0')
  o._element.spPr.append(OxmlElement('a:effectLst'))
  connector_records.append((o, i==0, i==len(pts)-2, (x1,y1),(x2,y2)))
  o.line.color.rgb=rgb(col);o.line.width=Pt(1.3 if kind=='flow' else 1.1)
  if kind=='supervision':o.line.dash_style=MSO_LINE_DASH_STYLE.ROUND_DOT
  if kind=='transfer':o.line.dash_style=MSO_LINE_DASH_STYLE.DASH
  if i==len(pts)-2 and arrow:
   end=OxmlElement('a:tailEnd');end.set('type','triangle');end.set('w','sm');end.set('len','sm');o._element.spPr.get_or_add_ln().append(end)
 edges.append({'name':name,'kind':kind,'points':pts})
 return g

def feature(sh,x,y,label,color='vision',size=.44):
 o=shape(sh,x-size/2,y-size/2,size,size,color,'feat',MSO_SHAPE.OVAL,'Feature_'+label)
 if label in ('S_Q','S_A','sG','sL','sP','g1','g2'):
  base,sub={'S_Q':('S','Q'),'S_A':('S','A'),'sG':('s','G'),'sL':('s','L'),'sP':('s','P'),'g1':('g','1'),'g2':('g','2')}[label]
  t=txt(sh,x-size/2,y-size/2,size,size,base+sub,20,True,name='FeatureLabel_'+label)
  tf=t.text_frame;tf.clear();p=tf.paragraphs[0];p.alignment=PP_ALIGN.CENTER
  for value,fs,baseline in ((base,20,0),(sub,13,-20000)):
   r=p.add_run();r.text=value;r.font.name=FONT;r.font.size=Pt(fs);r.font.bold=True;r.font.color.rgb=rgb('ink')
   if baseline:r._r.get_or_add_rPr().set('baseline',str(baseline))
 else:
  txt(sh,x-size/2,y-size/2,size,size,label,20,True,name='FeatureLabel_'+label)
 return o

def op(sh,x,y,label,size=.42,fill='white'):
 o=shape(sh,x-size/2,y-size/2,size,size,fill,'line',MSO_SHAPE.OVAL,'Operator_'+label)
 txt(sh,x-size/2,y-size/2,size,size,label,23,True,name='OperatorLabel_'+label)
 return o

def pool(sh,x,y):
 shape(sh,x-.29,y-.30,.58,.60,'qmod','line',MSO_SHAPE.ROUNDED_RECTANGLE,'Weighted_token_pool')
 txt(sh,x-.29,y-.30,.58,.60,'P',23,True,name='PoolLabel')

def loss(sh,x,y,w,h,label,fill='white',size=21):
 o=block(sh,x,y,w,h,label,fill=fill,stroke='train',size=size)
 o[4].line.dash_style=MSO_LINE_DASH_STYLE.ROUND_DOT
 return o

def lock(sh,x,y):
 g=group(sh,'Frozen_text_parameters_lock')
 shape(g.shapes,x+.045,y,.15,.20,'text','ink',MSO_SHAPE.OVAL,'Lock_shackle')
 shape(g.shapes,x,y+.11,.24,.20,'text','ink',MSO_SHAPE.RECTANGLE,'Lock_body')

def tokens(sh,x,y,w=1.45):
 g=group(sh,'Five_trainable_context_tokens')
 for j in range(5):shape(g.shapes,x+j*w/5,y,w/5-.055,.20,'text','tb',MSO_SHAPE.RECTANGLE,f'Context_token_{j+1}')
 return g

def image_tag(sh,x,y,label,w=1.05,h=.64):
 g=group(sh,'Input_'+label)
 shape(g.shapes,x,y,w,h,'white','border',MSO_SHAPE.RECTANGLE,'Image_frame_'+label)
 # All input placeholders are also editable vector objects, not bitmaps.
 shape(g.shapes,x+.07,y+.08,.14,.14,'vision',None,MSO_SHAPE.OVAL,'Image_sun')
 shape(g.shapes,x+.09,y+.28,.34,.25,'vision',None,MSO_SHAPE.ISOSCELES_TRIANGLE,'Image_peak')
 txt(g.shapes,x+.44,y+.06,w-.47,h-.1,label,22,True,name='Image_label')


def build(out):
 global P,S
 P=Presentation();P.slide_width=Inches(W);P.slide_height=Inches(H)
 S=P.slides.add_slide(P.slide_layouts[6]);S.background.fill.solid();S.background.fill.fore_color.rgb=rgb('white')
 a=group(S.shapes,'Panel_A_Pretraining');b=group(S.shapes,'Panel_B_Downstream_Assessment');lg=group(S.shapes,'Legend_and_Transfer')
 A=a.shapes;B=b.shapes;L=lg.shapes
 shape(A,.23,.18,11.35,12.82,'white','border',name='Pretraining_panel')
 shape(B,11.95,.18,13.82,12.82,'white','border',name='Downstream_panel')
 txt(A,.52,.29,10.7,.55,'(a)  Pretraining',32,True,align=PP_ALIGN.LEFT,name='Pretraining_title')
 txt(B,12.25,.29,12.6,.55,'(b)  Downstream assessment',32,True,align=PP_ALIGN.LEFT,name='Downstream_title')

 # A1: two image-only self-supervised views; repeated V/G/PI share parameters within stage.
 txt(A,.58,1.04,10.5,.34,'SAFEAUG VIEWS  ·  shared weights: V, G, PI, Pred I',20,bold=True,color='muted',align=PP_ALIGN.LEFT,name='Shared_I_I_IDs')
 for row,y in enumerate((1.66,2.91),1):
  cy=y+.39
  image_tag(A,.58,y+.07,'I'+str(row),1.0,.62)
  block(A,1.9,y,1.25,.78,'CLIP\nV','V','vision',size=22)
  block(A,3.47,y,1.12,.78,'Global\nG','G',size=21)
  feature(A,5.0,cy,'g'+str(row))
  block(A,5.57,y,1.05,.78,'PI\nnorm','PI',size=22)
  dot(A,6.90,cy);txt(A,6.72,y+.50,.6,.36,'v'+str(row),21,italic=True,name='PI_output')
  block(A,7.24,y,1.09,.78,'Pred I',fill='proj',size=21)
  edge(A,[(1.58,cy),(1.9,cy)],name=f'I{row}_to_V')
  edge(A,[(3.15,cy),(3.47,cy)],name=f'V{row}_to_G')
  edge(A,[(4.59,cy),(4.78,cy)],name=f'G{row}_to_g')
  edge(A,[(5.22,cy),(5.57,cy)],name=f'g{row}_to_PI')
  edge(A,[(6.62,cy),(7.24,cy)],name=f'PI{row}_to_predictor')
  edge(A,[(8.33,cy),(9.17,cy)],'supervision',name=f'p{row}_SimSiam')
 # sg targets are explicit separate branches, not parameter-sharing dashes.
 edge(A,[(6.90,2.05),(6.90,1.45),(10.25,1.45),(10.25,1.68)],'supervision',name='sg_target_v1')
 edge(A,[(6.90,3.30),(6.90,3.95),(10.25,3.95),(10.25,3.70)],'supervision',name='sg_target_v2')
 loss(A,9.17,1.68,2.10,2.02,'Symmetric\ncross-view I–I\n\nsg targets',size=22)
 txt(A,.58,4.13,10.45,.32,'CLEAN VIEW  ·  human quality / alignment / component labels',20,True,color='muted',align=PP_ALIGN.LEFT,name='Clean_view_caption')

 # A2 clean visual stem and expanded-in-B branch macros.
 image_tag(A,.60,5.46,'Iᶜ',1.02,.88)
 dot(A,1.77,5.90)
 edge(A,[(1.62,5.90),(1.77,5.90)],name='Clean_input')
 for yy,enc,proj,flabel in ((4.82,'V','G','g'),(6.32,'R','L','P₀')):
  cy=yy+.38
  edge(A,[(1.77,5.90),(1.77,cy),(2.10,cy)],name='Clean_to_'+enc)
  block(A,2.10,yy,1.25,.76,'CLIP · V' if enc=='V' else 'ResNet\nR',enc,'vision',size=21)
  block(A,3.70,yy,1.02,.76,'Global\nG' if proj=='G' else 'Local\nL',proj,size=21)
  edge(A,[(3.35,cy),(3.70,cy)],name=enc+'_to_'+proj)
  edge(A,[(4.72,cy),(5.04,cy)],name='Clean_'+flabel)
  feature(A,5.26,cy,flabel)
 block(A,6.01,4.80,2.36,1.14,'Quality branch\nsee (b)',fill='qmod',stroke='qb',size=23)
 edge(A,[(5.48,5.20),(6.01,5.20)],name='g_to_Q_clean')
 edge(A,[(5.26,6.48),(5.26,6.13),(6.55,6.13),(6.55,5.94)],name='P0_to_Q_clean')
 feature(A,8.73,5.20,'S_Q','qmod',.53)
 edge(A,[(8.37,5.20),(8.46,5.20)],name='Clean_quality_score')
 loss(A,9.20,4.77,2.05,.89,'MOS labels\nReg. + rank','q',21)
 edge(A,[(8.995,5.20),(9.20,5.20)],'supervision',name='Q_human_supervision')
 feature(A,8.73,6.23,'q','qmod',.47)
 edge(A,[(8.05,5.94),(8.05,6.23),(8.495,6.23)],name='Quality_representation')
 loss(A,9.20,5.88,2.05,.95,'Components\nAnchor / ordinal','q',20)
 edge(A,[(8.965,6.23),(9.20,6.23)],'supervision',name='Component_anchor_supervision')
 block(A,6.01,6.65,2.36,.64,'H_art  /  H_aes',fill='qmod',stroke='qb',size=23,head=True)
 edge(A,[(7.15,5.94),(7.15,6.65)],name='Q_features_to_component_heads')
 txt(A,5.98,6.23,1.55,.29,'qA, qL',18,color='muted',name='Auxiliary_head_inputs')
 loss(A,9.20,6.99,2.05,.63,'Aux. reg. / rank','q',19)
 edge(A,[(8.37,6.97),(8.80,6.97),(8.80,7.30),(9.20,7.30)],'supervision',name='Component_aux_supervision')
 block(A,6.01,7.90,2.36,.94,'Alignment branch\nsee (b)',fill='amod',stroke='ab',size=22)
 # g reused for alignment; two separate feature buses.
 dot(A,5.65,5.20)
 edge(A,[(5.48,5.20),(5.65,5.20),(5.65,8.16),(6.01,8.16)],name='Clean_g_reuse_for_align')
 dot(A,5.26,6.70)
 edge(A,[(5.26,6.92),(5.26,8.57),(6.01,8.57)],name='Clean_P0_reuse_for_align')
 feature(A,8.73,8.25,'S_A','amod',.53)
 edge(A,[(8.37,8.25),(8.465,8.25)],name='Clean_alignment_score')
 loss(A,9.20,7.92,2.05,.86,'Align labels\nReg. / rank / SR','a',20)
 edge(A,[(8.995,8.25),(9.20,8.25)],'supervision',name='Align_human_supervision')

 # A3 text path. Prompt and answer route differences explicitly documented.
 txt(A,.58,9.10,4.30,.30,'TEXT · shared Eₜ, T, PT',20,True,color='muted',align=PP_ALIGN.LEFT,name='Text_shared_IDs')
 block(A,.58,9.66,1.34,.83,'Prompt /\nAnswer',fill='text',stroke='tb',size=21)
 block(A,2.20,9.66,1.45,.83,'CLIP text\nEₜ',id='ET',fill='text',stroke='tb',size=22)
 lock(A,3.32,9.71)
 tokens(A,2.22,10.57,1.42)
 edge(A,[(2.93,10.57),(2.93,10.49)],name='Context_A_into_text_encoder')
 txt(A,1.97,10.80,2.05,.61,'5 trainable\ncontext tokens',18,color='muted',name='Trainable_context_A')
 block(A,3.97,9.66,1.00,.83,'Text T',id='T',fill='text',stroke='tb',size=22)
 block(A,5.30,9.66,1.00,.83,'PT\nnorm',id='PT',fill='text',stroke='tb',size=22)
 block(A,6.64,9.66,1.65,.83,'Text routes†',fill='text',stroke='tb',size=21)
 for x1,x2,name in ((1.92,2.20,'Input_text'),(3.65,3.97,'ET_to_T'),(4.97,5.30,'T_to_PT'),(6.30,6.64,'PT_to_text_routes')):
  edge(A,[(x1,10.075),(x2,10.075)],name=name)
 loss(A,9.20,9.36,2.05,1.43,'I–Prompt\nI–Answer\nPrompt–Answer','a',21)
 edge(A,[(8.29,10.075),(9.20,10.075)],'supervision',name='Text_relation_losses')
 edge(A,[(7.65,8.84),(7.65,9.08),(8.84,9.08),(8.84,9.65),(9.20,9.65)],'supervision',name='Reuse_clean_semantic_image_feature')

 # text_fused enters clean Align before PT; complete branch expanded in B.
 dot(A,5.12,10.075)
 edge(A,[(4.97,10.075),(5.12,10.075),(5.12,9.18),(6.70,9.18),(6.70,8.84)],name='Text_T_to_clean_alignment')
 txt(A,.60,11.46,10.47,.48,'† Prompt: PT → AT → Wₜ → norm;  Answer: PT → norm.',20,align=PP_ALIGN.LEFT,name='Exact_prompt_answer_routes')
 txt(A,.60,12.33,9.50,.43,'Positive local / word supervision: RichHF annotations.',20,color='muted',align=PP_ALIGN.LEFT,name='Pretrain_local_and_anchor_note')

 # B background task bands.
 shape(B,17.67,1.12,7.78,5.45,'q','qb',name='Quality_branch_region')
 shape(B,17.67,7.45,7.78,5.03,'a','ab',name='Alignment_branch_region')
 txt(B,17.91,1.22,3.7,.42,'Quality branch',27,True,align=PP_ALIGN.LEFT,name='Quality_title')
 txt(B,17.91,7.56,4.5,.42,'Alignment branch',27,True,align=PP_ALIGN.LEFT,name='Alignment_title')

 # B shared visual stem and NSS, with explicit distinct projection modules.
 image_tag(B,12.20,3.55,'I',.86,.84)
 edge(B,[(13.06,3.97),(13.22,3.97)],name='B_input_stem');dot(B,13.22,3.97)
 block(B,13.54,1.98,1.52,.84,'CLIP\nV','V','vision',size=23)
 block(B,15.42,1.98,1.12,.84,'Global\nG','G',size=22)
 block(B,13.54,3.69,1.52,.80,'NSS · 6D',fill='qmod',stroke='qb',size=22)
 block(B,15.42,3.69,1.12,.80,'NSS\nMLP',fill='qmod',stroke='qb',size=22)
 block(B,13.54,5.25,1.52,.84,'ResNet\nlayer3 · R','R','vision',size=22)
 block(B,15.42,5.25,1.12,.84,'Local\nproj. · L','L',size=22)
 for cy,xend,name in ((2.40,13.54,'I_to_global_encoder'),(4.09,13.54,'I_to_NSS'),(5.67,13.54,'I_to_local_encoder')):
  edge(B,[(13.22,3.97),(13.22,cy),(xend,cy)],name=name)
 for cy,name in ((2.40,'V_to_G'),(4.09,'NSS_to_MLP'),(5.67,'R_to_L')):edge(B,[(15.06,cy),(15.42,cy)],name=name)
 feature(B,17.08,2.40,'g');feature(B,17.08,5.67,'P₀')
 edge(B,[(16.54,2.40),(16.86,2.40)],name='Global_feature_g')
 edge(B,[(16.54,5.67),(16.86,5.67)],name='Local_tokens_P0')
 txt(B,15.10,2.92,1.85,.30,'768 → 256',20,color='muted',name='Global_dimension')
 txt(B,13.45,6.20,3.0,.30,'P₀: 196 × 256',20,color='muted',name='Local_dimension')
 txt(B,13.35,6.66,3.10,.33,'L: two 1×1 projections',18,color='muted',name='Two_local_projections')

 # B Quality global residual; g bypass is the original global vector, not PI.
 block(B,18.10,2.00,1.24,.80,'QA\nnorm',id='QA',fill='qmod',stroke='qb',size=23)
 op(B,20.23,2.40,'+')
 block(B,20.90,2.00,1.25,.80,'H_QG',id='HQG',fill='qmod',stroke='qb',head=True,size=23)
 feature(B,22.67,2.40,'sG','qmod',.46)
 edge(B,[(17.30,2.40),(18.10,2.40)],name='g_to_QA')
 edge(B,[(19.34,2.40),(20.02,2.40)],name='QA_residual_direction')
 txt(B,19.12,2.95,2.00,.40,'α·sg(‖g‖)·qA',20,name='Alpha_residual_scale')
 edge(B,[(17.55,2.40),(17.55,1.77),(20.23,1.77),(20.23,2.19)],name='g_residual_skip')
 dot(B,17.55,2.40)
 edge(B,[(20.44,2.40),(20.90,2.40)],name='Quality_global_to_head')
 edge(B,[(22.15,2.40),(22.44,2.40)],name='Global_quality_score')
 # B local tokens and NSS are added, never g and P0.
 op(B,18.30,5.67,'+')
 block(B,18.90,5.26,1.20,.82,'QL\nnorm',id='QL',fill='qmod',stroke='qb',size=23)
 pool(B,20.68,5.67)
 block(B,21.27,5.26,1.20,.82,'H_QL',id='HQL',fill='qmod',stroke='qb',head=True,size=23)
 feature(B,22.98,5.67,'sL','qmod',.46)
 edge(B,[(17.30,5.67),(18.09,5.67)],name='P0_to_quality_local')
 edge(B,[(16.54,4.09),(18.30,4.09),(18.30,4.76)],name='NSS_embedding_upper',arrow=False)
 edge(B,[(18.30,4.88),(18.30,5.46)],name='NSS_embedding_to_local_add')
 edge(B,[(18.51,5.67),(18.90,5.67)],name='Local_sum_to_QL')
 edge(B,[(20.10,5.67),(20.39,5.67)],name='QL_to_pool')
 edge(B,[(20.97,5.67),(21.27,5.67)],name='Pool_to_local_quality_head')
 edge(B,[(22.47,5.67),(22.75,5.67)],name='Local_quality_score')
 dot(B,15.20,4.09)
 # raw NSS also enters the gate; it is not only embedded and added.
 edge(B,[(15.20,4.09),(15.20,4.82),(20.68,4.82),(20.68,5.37)],name='NSS_to_pool_gate')
 txt(B,19.34,4.28,2.83,.42,'NSS-gated pooling',20,color='muted',name='NSS_pool_annotation')
 # Score mixture is a different operator from residual addition.
 op(B,24.02,3.97,'M',.52,'qmod')
 edge(B,[(22.90,2.40),(24.02,2.40),(24.02,3.71)],name='Global_score_to_quality_mix')
 edge(B,[(23.21,5.67),(24.02,5.67),(24.02,4.23)],name='Local_score_to_quality_mix')
 edge(B,[(24.28,3.97),(24.72,3.97)],name='Quality_output')
 feature(B,25.00,3.97,'S_Q','qmod',.56)
 txt(B,24.30,3.13,1.10,.40,'Quality',25,True,name='Quality_output_title')
 txt(B,21.15,3.52,2.20,.75,'(1−β)sG\n+ βsL',22,name='Quality_score_mix_formula')

 # Feature buses into Alignment. Crossings are drawn with a physical gap, no joint.
 # g bus: vertical portions leave gaps where unrelated horizontal NSS wires pass.
 edge(B,[(17.08,2.62),(17.08,3.99)],name='g_bus_upper',arrow=False)
 edge(B,[(17.08,4.19),(17.08,4.72)],name='g_bus_middle',arrow=False)
 edge(B,[(17.08,4.92),(17.08,5.36)],name='g_bus_lower',arrow=False)
 # Avoid P0 circular node: jog to the left of its source, no accidental junction.
 edge(B,[(17.08,5.36),(16.72,5.36),(16.72,5.60)],name='g_jog_above_P0',arrow=False)
 edge(B,[(16.72,5.74),(16.72,8.49),(17.46,8.49)],name='g_bus_gap_before_P0_bus',arrow=False)
 edge(B,[(17.58,8.49),(18.10,8.49)],name='g_bus_to_PI')
 # P0 branch uses its own bus to AP; no PI in this path.
 dot(B,17.52,5.67)
 edge(B,[(17.30,5.67),(17.52,5.67),(17.52,10.12),(18.10,10.12)],name='P0_to_alignment_adapter')

 block(B,18.10,8.08,1.16,.82,'PI\nnorm','PI',size=23)
 block(B,19.61,8.08,1.15,.82,'AG\nadapter',id='AG',fill='amod',stroke='ab',size=24)
 edge(B,[(19.26,8.49),(19.61,8.49)],name='PI_to_AG')
 dot(B,21.03,8.49);edge(B,[(20.76,8.49),(21.43,8.49)],name='AG_to_global_pair')
 op(B,21.64,8.49,'C',.42,'amod')
 block(B,22.12,8.08,1.14,.82,'H_AG',id='HAG',fill='amod',stroke='ab',head=True,size=23)
 edge(B,[(21.85,8.49),(22.12,8.49)],name='Global_pair_to_head')
 feature(B,23.66,8.49,'sG','amod',.44)
 edge(B,[(23.26,8.49),(23.44,8.49)],name='Global_alignment_score')

 block(B,18.10,9.71,1.16,.82,'AP\nadapter',id='AP',fill='amod',stroke='ab',size=24)
 block(B,19.61,9.71,1.15,.82,'Text\npool',fill='amod',stroke='ab',size=19)
 edge(B,[(19.26,10.12),(19.61,10.12)],name='AP_to_text_pool')
 op(B,21.64,10.55,'C',.42,'amod')
 edge(B,[(20.76,10.12),(21.15,10.12),(21.15,10.55),(21.43,10.55)],name='Local_evidence_to_pair')
 block(B,22.12,10.14,1.14,.82,'H_AP',id='HAP',fill='amod',stroke='ab',head=True,size=23)
 edge(B,[(21.85,10.55),(22.12,10.55)],name='Patch_pair_to_head')
 feature(B,23.66,10.55,'sP','amod',.44)
 edge(B,[(23.26,10.55),(23.44,10.55)],name='Patch_alignment_score')

 # Text stem and adapters, separately trainable context.
 block(B,12.20,11.32,.99,.83,'Prompt',fill='text',stroke='tb',size=20)
 block(B,13.54,11.32,1.52,.83,'CLIP text\nEₜ','ET','text','tb',size=22)
 lock(B,14.73,11.36)
 block(B,15.42,11.32,1.12,.83,'Text T','T','text','tb',size=22)
 tokens(B,13.56,10.72,1.47)
 edge(B,[(14.295,10.92),(14.295,11.32)],name='Context_B_into_text_encoder')
 txt(B,13.30,10.04,2.10,.61,'5 trainable\ncontext tokens',18,color='muted',name='Trainable_context_B')
 block(B,18.10,11.32,1.16,.83,'PT\nnorm','PT','text','tb',size=23)
 block(B,19.61,11.32,1.15,.83,'AT\nadapter',id='AT',fill='amod',stroke='ab',size=24)
 for x1,x2,name in ((13.19,13.54,'Prompt_to_ET'),(15.06,15.42,'ET_to_T_B'),(16.54,18.10,'T_to_PT_B'),(19.26,19.61,'PT_to_AT')):
  edge(B,[(x1,11.735),(x2,11.735)],name=name)
 # aT forks below the pool, so it does not join the pooled image evidence.
 dot(B,21.02,11.735)
 edge(B,[(20.76,11.735),(21.02,11.735),(21.02,11.03),(21.36,11.03),(21.36,10.62)],name='aT_global_bus_lower',arrow=False)
 edge(B,[(21.36,10.49),(21.36,9.14),(21.64,9.14),(21.64,8.70)],name='Text_to_global_pair')
 dot(B,21.02,11.03);dot(B,21.36,11.03)
 edge(B,[(21.36,11.03),(21.64,11.03),(21.64,10.76)],name='Text_to_local_pair')
 edge(B,[(21.02,11.03),(20.20,11.03),(20.20,10.53)],name='Text_controls_patch_pool')
 # Adaptive score gate takes image/text features AND the two scores.
 block(B,23.40,9.14,1.05,.90,'Gate γ\nM',fill='amod',stroke='ab',size=22)
 edge(B,[(23.88,8.49),(23.95,8.49),(23.95,9.14)],name='Global_score_to_align_gate')
 edge(B,[(23.88,10.55),(23.95,10.55),(23.95,10.04)],name='Local_score_to_align_gate')
 edge(B,[(21.03,8.49),(21.03,8.00),(24.30,8.00),(24.30,9.14)],name='Global_feature_to_align_gate')
 edge(B,[(21.02,11.735),(21.02,12.23),(24.30,12.23),(24.30,10.62)],name='Text_feature_to_align_gate_lower',arrow=False)
 edge(B,[(24.30,10.49),(24.30,10.04)],name='Text_feature_to_align_gate')
 edge(B,[(24.45,9.59),(24.72,9.59)],name='Alignment_output')
 feature(B,25.00,9.59,'S_A','amod',.56)
 txt(B,21.93,9.08,1.40,.79,'γsG +\n(1−γ)sP',22,name='Align_mix_formula')
 txt(B,24.40,8.86,1.24,.35,'Align',21,True,name='Alignment_output_title')

 txt(A,.60,11.91,10.20,.40,'q = norm[(1−ρ)qA + ρqL] → fixed component anchors',21,align=PP_ALIGN.LEFT,name='Quality_embedding_formula')
 # Parameter migration is ONE explicit bundle, not feature sharing or reuse.
 block(L,10.38,12.37,.86,.44,'θᵣ',fill='text',stroke='transfer',size=21)
 block(L,12.24,12.37,.86,.44,'θᵣ',fill='text',stroke='transfer',size=21)
 edge(L,[(11.24,12.59),(12.24,12.59)],'transfer',name='Representation_parameter_initialization')
 txt(L,11.23,12.04,1.04,.28,'initialize',16,color='transfer',name='Transfer_label')
 txt(B,17.85,12.63,7.47,.28,'Inherit representations; reset task heads / score gate.',18,color='muted',name='Downstream_reset_note')

 # Bottom legend; training-specific data flow remains solid, losses alone dotted.
 shape(L,.25,13.18,25.50,1.23,'white','border',name='Legend_panel')
 edge(L,[(.56,13.53),(1.11,13.53)],name='Legend_forward')
 txt(L,1.22,13.31,2.20,.43,'Forward flow',21,align=PP_ALIGN.LEFT,name='Legend_flow_label')
 edge(L,[(3.55,13.53),(4.10,13.53)],'supervision',name='Legend_supervision')
 txt(L,4.21,13.31,3.42,.43,'Training supervision',21,align=PP_ALIGN.LEFT,name='Legend_supervision_label')
 edge(L,[(7.99,13.53),(8.54,13.53)],'transfer',name='Legend_transfer')
 txt(L,8.65,13.31,3.67,.43,'Parameter initialization',21,align=PP_ALIGN.LEFT,name='Legend_transfer_label')
 txt(L,12.55,13.31,9.49,.43,'Repeated module IDs = shared weights within each stage',21,align=PP_ALIGN.LEFT,name='Legend_shared_parameters')
 lock(L,23.02,13.37);txt(L,23.36,13.31,2.0,.43,'Frozen weights',21,align=PP_ALIGN.LEFT,name='Legend_frozen')
 op(L,.77,14.05,'+',.32);txt(L,1.04,13.86,2.28,.36,'Residual add',20,align=PP_ALIGN.LEFT,name='Legend_residual')
 op(L,3.59,14.05,'C',.32);txt(L,3.86,13.86,4.57,.36,'Pair concat. [u,t,|u−t|]',20,align=PP_ALIGN.LEFT,name='Legend_concat')
 pool(L,8.73,14.05);txt(L,9.16,13.86,2.61,.36,'Token pooling',20,align=PP_ALIGN.LEFT,name='Legend_pool')
 op(L,12.14,14.05,'M',.32);txt(L,12.43,13.86,2.25,.36,'Score mixture',20,align=PP_ALIGN.LEFT,name='Legend_score_mix')
 dot(L,15.15,14.05,.055);txt(L,15.39,13.86,2.50,.36,'Feature reuse',20,align=PP_ALIGN.LEFT,name='Legend_reuse')
 txt(L,18.30,13.86,3.20,.36,'sg: stop-gradient',20,align=PP_ALIGN.LEFT,name='Legend_stop_gradient')
 txt(L,21.35,13.86,4.0,.36,'PI/PT: projectors; H_*: heads',20,align=PP_ALIGN.LEFT,name='Legend_task_colors')

 # Attach native connectors to real editable shape ports wherever endpoints coincide.
 for conn,first,last,p0,p1 in connector_records:
  for attach,point in ((conn.begin_connect if first else None,p0),(conn.end_connect if last else None,p1)):
   if attach is None:continue
   found=False
   for target in anchor_shapes:
    x,y,w,h=[v/914400 for v in (target.left,target.top,target.width,target.height)]
    for index,port in enumerate(((x+w/2,y),(x,y+h/2),(x+w/2,y+h),(x+w,y+h/2))):
     if abs(point[0]-port[0])+abs(point[1]-port[1])<.006:
      attach(target,index);found=True;break
    if found:break
 pptx=out/'projection_only_v1_framework.pptx';P.save(pptx)
 # Speaker notes retain exact details without filling the publication diagram.
 S.notes_slide.notes_text_frame.text = (
  'Source: MYWZX/simsiam_w work, projection_only_v1. '+
  'G=semantic_proj; L includes local encoder 1x1 and independent local_patch_proj. '+
  'QA/PI/PT: 256-512-512-256 MLP, normalized where marked. '+
  'Quality residual hG=g+sigmoid(alpha_logit)*stopgrad(norm(g))*normalize(QA(g)); '+
  'QL(x)=x+0.1*MLP(LN(x)), followed by normalization. '+
  'Pool uses softmax GateNSS([QL(P0+MLPNSS(n)),n]); local score is predicted AFTER pooling. '+
  'q=normalize((1-rho)*qA+rho*qL); component anchors use fixed orthogonal planes; CE and ordinal BCE are ON. '+
  'Quality text-bank CE and negative push are OFF. '+
  'Align pool uses W_I(AP(P0)) and W_T(AT(norm(PT(T(E_text(Prompt)))))); keys/queries normalized, softmax over tokens; no cross-attention module. '+
  'Each C builds [image,text,abs(image-text)]. Each H_* includes sigmoid scalar output. '+
  'Align gate is conditioned on aG,aT,sG,sP. '+
  'Pretraining Prompt relations use W_T(AT(norm(PT(T(E_text(Prompt)))))); Answer relations use norm(PT(T(E_text(Answer)))). '+
  'θrepr: visual encoders V/R, global/local projections G/L, 5 trainable context tokens, T, PI/PT, QA, QL, NSS modules, AG/AP/AT, alpha,rho. '+
  'Downstream task heads, align patch matching projections W_I/W_T, alignment score gate, and score beta are reset; beta starts at 0.5. '+
  'Original frozen CLIP text weights are loaded from the same original source, not learned during pretraining. '+
  'Crossings with gaps/no junction dots are not connections. This is a consolidated architecture diagram; Clean branches are expanded in panel (b).')
 P.save(pptx)
 return pptx


def audit_and_render(pptx,out):
 executable=shutil.which('soffice') or shutil.which('libreoffice')
 if not executable:raise RuntimeError('LibreOffice/soffice is required for same-PPT rendering')
 subprocess.run([executable,'--headless','--convert-to','pdf','--outdir',str(out),str(pptx)],check=True,timeout=120)
 pdf=pptx.with_suffix('.pdf')
 if not pdf.exists():raise RuntimeError('LibreOffice did not produce PDF')
 doc=fitz.open(pdf)
 if len(doc)!=1:raise RuntimeError('Expected one slide / one PDF page')
 page=doc[0];scale=5600/page.rect.width
 pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
 png=out/'projection_only_v1_framework.png';pix.save(png)
 page.get_pixmap(matrix=fitz.Matrix(1800/page.rect.width,1800/page.rect.width),alpha=False).save(out/'preview.png')
 (out/'rendered_text.txt').write_text(page.get_text(),encoding='utf-8')
 check=Presentation(pptx);counts={'shapes':0,'connectors':0,'text_objects':0,'pictures':0,'groups':0}
 def walk(shapes):
  for s in shapes:
   counts['shapes']+=1
   if s.shape_type==6:counts['groups']+=1;walk(s.shapes)
   if s.shape_type==9:counts['connectors']+=1
   if s.shape_type==13:counts['pictures']+=1
   if s.has_text_frame and s.text:counts['text_objects']+=1
 walk(check.slides[0].shapes)
 # Content coverage against text exported from the actual PDF.
 text=page.get_text();required=['Pretraining','Downstream assessment','Quality branch','Alignment branch','H_QG','H_QL','H_AG','H_AP','Anchor / ordinal','5 trainable context tokens']
 normalized=' '.join(text.split())
 missing=[w for w in required if w not in normalized]
 if missing:raise RuntimeError('Missing rendered text: '+repr(missing))
 fonts=sorted(set(f[3] for f in page.get_fonts()))
 report={'canvas_inches':[W,H],'png_pixels':[pix.width,pix.height],'pdf_pages':len(doc),'font_family':FONT,'pdf_fonts':fonts,'editable_objects':counts,'required_render_text_present':True,'font_fit_adjustments':fitted,'anchored_native_connector_endpoints':sum(len(o._element.findall('.//'+qn('a:stCxn')))+len(o._element.findall('.//'+qn('a:endCxn'))) for o,_,_,_,_ in connector_records),'content_source':'projection_only_v1_framework.pptx','rasterized_objects':'None; all drawing objects including input placeholders are native editable shapes.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (pptx,pdf,png)}}
 (out/'render_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
 args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 p=build(out);audit_and_render(p,out)
