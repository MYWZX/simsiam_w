#!/usr/bin/env python3
"""Native editable QAM/AAM/SRM figure set; render PDF/PNG from each PPTX."""
from pathlib import Path
import argparse,json,subprocess,shutil,hashlib
import fitz
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.enum.text import PP_ALIGN
import drawing_primitives as d
HERE=Path(__file__).resolve().parent
def canvas(w,h):
 p=Presentation();p.slide_width=Inches(w);p.slide_height=Inches(h)
 s=p.slides.add_slide(p.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white')
 return p,s

def label(sh,x,y,w,h,text,size=24,bold=False,color='ink',name='label'):
 return d.txt(sh,x,y,w,h,text,size,bold,color,PP_ALIGN.CENTER,name)

def anchor_native_connections(p):
 from pptx.enum.shapes import MSO_SHAPE_TYPE
 items=[]
 def walk(shapes):
  for z in shapes:
   if z.shape_type==6:walk(z.shapes)
   else:items.append(z)
 walk(p.slides[0].shapes)
 targets=[z for z in items if z.shape_type==MSO_SHAPE_TYPE.AUTO_SHAPE and .3<z.width/914400<7 and .3<z.height/914400<4]
 for z in items:
  if z.shape_type!=MSO_SHAPE_TYPE.LINE:continue
  for attach,px,py in ((z.begin_connect,z.begin_x,z.begin_y),(z.end_connect,z.end_x,z.end_y)):
   found=False
   for t in targets:
    x,y,w,h=t.left,t.top,t.width,t.height
    for idx,(xx,yy) in enumerate(((x+w/2,y),(x,y+h/2),(x+w/2,y+h),(x+w,y+h/2))):
     if abs(px-xx)+abs(py-yy)<4500:attach(t,idx);found=True;break
    if found:break



def stem(sh,x,y,stage):
 g=d.group(sh,'Shared_representation_stem_'+stage).shapes
 d.shape(g,x,y,2.40,7.40,'white','feat',name='Shared_stem_body_'+stage,width=1.6)
 label(g,x+.12,y+.28,2.16,1.0,'Shared\nencoding',28,True)
 for off,text,fill in [(1.65,'CLIP V · G','vision'),(2.70,'ResNet R · L','vision'),(3.75,'Text Eₜ · T','text'),(4.80,'PI / PT','proj'),(5.85,'NSS n','qmod')]:
  d.block(g,x+.20,y+off,2.0,.70,text,fill=fill,size=22)
 label(g,x+.12,y+6.72,2.16,.62,'Frozen Eₜ\n5 trainable ctx',19,color='muted')

def task(sh,x,y,which):
 name='Quality' if which=='QAM' else 'Alignment';fill='qmod' if which=='QAM' else 'amod';stroke='qb' if which=='QAM' else 'ab'
 h=1.80 if which=='QAM' else 2.50;g=d.group(sh,which+'_encapsulated').shapes
 d.shape(g,x,y,2.30,h,fill,stroke,name=which+'_body',width=1.6)
 label(g,x+.10,y+.15,2.10,.48,which,31,True)
 label(g,x+.10,y+.66,2.10,.66 if which=='QAM' else 1.00,name+'\nassessment',21)
 label(g,x+.08,y+h-.35,2.14,.26,'score S_Q' if which=='QAM' else 'score S_A',17,color='muted')

def overview(out):
 p,s=canvas(24,13.30);A=d.group(s.shapes,'Pretraining_QAM_AAM').shapes;B=d.group(s.shapes,'Downstream_QAM_AAM').shapes;L=d.group(s.shapes,'Transfer_and_legend').shapes
 d.shape(A,.22,.22,15.30,10.20,'white','border',name='Pretraining_panel');d.shape(B,15.86,.22,7.92,10.20,'white','border',name='Downstream_panel')
 d.txt(A,.55,.49,14,.60,'(a) Pretraining',33,True,align=PP_ALIGN.LEFT)
 d.txt(B,16.12,.49,7.25,.60,'(b) Downstream assessment',30,True,align=PP_ALIGN.LEFT)
 label(A,.60,1.30,7.9,.4,'Shared representations · separate task modules',23,color='muted');label(A,10.6,1.3,4.4,.4,'Four primary SRM relations',22,color='muted')
 stem(A,3.10,1.95,'pretrain');task(A,6.0,7.30,'QAM');task(A,6.0,3.75,'AAM')
 for yy,tx in [(2.26,'SafeAug I₁'),(3.36,'SafeAug I₂'),(4.76,'Clean Iᶜ')]:
  d.image_tag(A,.55,yy,tx,1.98,.68);d.edge(A,[(2.53,yy+.34),(3.10,yy+.34)],name='Image_'+tx)
 for yy,tx in [(6.22,'Prompt P'),(7.62,'Answer A')]:
  d.block(A,.55,yy,1.98,.76,tx,fill='text',stroke='tb',size=24);d.edge(A,[(2.53,yy+.38),(3.1,yy+.38)],name='Text_'+tx)
 label(A,.56,8.67,2.02,.66,'Same weights\nacross views',21,color='muted')
 label(A,5.65,3.40,2.60,.30,'in: vᶜ, P₀, tP⁰',19,color='muted')
 label(A,5.65,6.95,2.60,.30,'in: g, P₀, n',20,color='muted')
 # Image-image PI features bypass both assessment modules.
 for yy,lab in [(2.56,'v₁'),(3.18,'v₂')]:
  d.edge(A,[(5.50,yy),(8.50,yy)],name='Stem_'+lab);d.feature(A,8.75,yy,lab,'vision',.50);d.edge(A,[(9.0,yy),(10.80,yy)],name=lab+'_to_II')
 # Clean image and prompt representations enter AAM; raw quality inputs bypass PI.
 d.edge(A,[(5.50,4.98),(6.00,4.98)],name='vC_P0_t0_to_AAM')
 d.edge(A,[(5.50,8.05),(6.00,8.05)],name='g_P0_n_to_QAM')
 for yy,lab,fill in [(4.35,'aᶜ','vision'),(5.02,'tP','text')]:
  d.edge(A,[(8.30,yy),(8.50,yy)],name='AAM_'+lab);d.feature(A,8.75,yy,lab,fill,.50)
 d.edge(A,[(5.50,6.65),(8.15,6.65),(8.15,6.45),(8.50,6.45)],name='Answer_PT_bypasses_AAM');d.feature(A,8.75,6.45,'tA','text',.50)
 # Literal tensor reuse; crossing gaps have no junction dots.
 d.edge(A,[(9.00,4.35),(10.80,4.35)],name='aC_to_IP');d.dot(A,9.25,4.35)
 d.edge(A,[(9.25,4.35),(9.25,4.96)],name='aC_branch1',arrow=False);d.edge(A,[(9.25,5.08),(9.25,6.05),(10.80,6.05)],name='aC_to_IA')
 d.edge(A,[(9.00,5.02),(9.65,5.02),(9.65,4.85),(10.80,4.85)],name='tP_to_IP');d.dot(A,9.65,5.02)
 d.edge(A,[(9.65,5.02),(9.65,5.99)],name='tP_branch1',arrow=False);d.edge(A,[(9.65,6.11),(9.65,6.39)],name='tP_branch2',arrow=False);d.edge(A,[(9.65,6.51),(9.65,7.45),(10.80,7.45)],name='tP_to_PA')
 d.edge(A,[(9.00,6.45),(10.80,6.45)],name='tA_to_IA');d.dot(A,10.05,6.45)
 d.edge(A,[(10.05,6.45),(10.05,7.39)],name='tA_branch1',arrow=False);d.edge(A,[(10.05,7.51),(10.05,8.02),(10.80,8.02)],name='tA_to_PA')
 for k,(yy,title,lo) in enumerate([(2.24,'I–I','L_II'),(4.02,'I–Prompt','L_IP'),(5.72,'I–Answer','L_IA'),(7.15,'Prompt–Answer','L_PA')],1):
  d.block(A,10.80,yy,2.35,1.20,f'SRM {k}\n{title}',fill='proj',stroke='feat',size=25);d.edge(A,[(13.15,yy+.60),(13.60,yy+.60)],'supervision',name=lo);d.block(A,13.60,yy+.27,1.2,.66,lo,stroke='train',size=23)
 d.block(A,10.8,9.02,4.0,.90,'Clean-view objectives\nScores / components / ordinal',fill='q',stroke='qb',size=22)
 d.edge(A,[(8.30,8.05),(8.65,8.05),(8.65,9.25),(10.80,9.25)],'supervision',name='Quality_Clean_supervision')
 d.edge(A,[(8.30,6.05),(8.43,6.05),(8.43,6.39)],'supervision',name='Align_score_upper',arrow=False)
 d.edge(A,[(8.43,6.51),(8.43,9.65),(10.80,9.65)],'supervision',name='Alignment_Clean_supervision')
 label(A,7.12,10.03,7.7,.27,'PCGrad · positive local / word supervision retained',19,color='muted')
 # Exactly the same two task module symbols are reused in the downstream panel.
 stem(B,16.25,1.95,'downstream');task(B,19.50,3.00,'QAM');task(B,19.50,6.40,'AAM')
 d.block(B,16.14,1.18,1.18,.55,'Image I',fill='vision',size=21);d.block(B,17.60,1.18,1.18,.55,'Prompt P',fill='text',stroke='tb',size=21)
 d.edge(B,[(16.73,1.73),(16.73,1.95)],name='Downstream_I');d.edge(B,[(18.19,1.73),(18.19,1.95)],name='Downstream_P')
 label(B,19.50,2.62,2.30,.30,'in: g, P₀, n',20,color='muted');label(B,19.50,6.02,2.30,.30,'in: v, P₀, t',20,color='muted')
 d.edge(B,[(18.65,3.80),(19.50,3.80)],name='Quality_inputs_g_P0_n');d.edge(B,[(18.65,7.65),(19.50,7.65)],name='Align_inputs_v_P0_t')
 for yy,tx,fill in [(3.80,'S_Q','qmod'),(7.65,'S_A','amod')]:
  d.edge(B,[(21.80,yy),(22.35,yy)],name='Task_output_'+tx);d.block(B,22.35,yy-.44,1.12,.88,tx,fill=fill,size=28)
 label(B,16.20,9.63,7.3,.35,'Shared stem · independent Quality / Alignment outputs',20,color='muted')
 d.block(L,4.00,10.90,1.90,.60,'θrepr (pre)',fill='text',stroke='transfer',size=24);d.block(L,19.12,10.90,1.90,.60,'θrepr (init)',fill='text',stroke='transfer',size=24)
 d.edge(L,[(5.9,11.2),(19.12,11.2)],'transfer',name='Representation_transfer_only')
 label(L,6.4,10.58,12.0,.38,'Transfer representation parameters; reset dataset heads / matching Linear layers / score gates',21,color='transfer')
 label(L,.55,11.72,22.9,.42,'QAM and AAM reuse the same shared stem. Cross-stage initialization does not tie the two stages’ weights.',22,color='muted')
 d.shape(L,.23,12.34,23.54,.70,'white','border',name='Legend')
 for x,k,tx in [(.6,'flow','Forward features'),(3.8,'supervision','Training supervision'),(8.1,'transfer','Parameter initialization')]:
  d.edge(L,[(x,12.69),(x+.5,12.69)],k,name='Legend_'+k);label(L,x+.58,12.46,2.8,.43,tx,20)
 d.dot(L,12.4,12.69,.055);label(L,12.58,12.46,2.6,.43,'Feature reuse',20);label(L,15.5,12.46,7.7,.43,'SRM 1–4 share h_I / h_T by input modality',21)
 anchor_native_connections(p);s.notes_slide.notes_text_frame.text='QAM and AAM are logical figure modules, not code changes. Shared encoding/projection stem contains V,R,G,L,ET,T,PI,PT and raw NSS descriptor n. QAM receives g,P0,n; no PI on Quality path. AAM receives v=normPI(g),P0,t=normPT(T(ET(P))). AAM exports normalized image relation aC and prompt relation tP. Answer tA bypasses AAM and comes directly from PT in the shared stem. I-I v1/v2 bypass QAM and AAM. SRM four relation pairs remain unchanged. All Clean labels, component anchors/ordinal, auxiliary heads, positive local/word losses and PCGrad remain. Theta repr includes inherited shared-stem parameters and task-representation adapters, not all task scoring/matching heads.'
 p.save(out/'01_split_framework.pptx')


def qam(out):
 p,s=canvas(16.5,12.3);G=d.group(s.shapes,'QAM_Quality_Assessment_Module').shapes
 label(G,.4,.28,15.7,.62,'QAM · Quality Assessment Module',32,True)
 d.shape(G,2.45,1.05,13.45,7.85,'q','qb',name='Quality_computation_region')
 for xx,yy,tx,si in [(1.0,2.35,'g',.64),(1.0,5.75,'P₀',.68),(1.0,7.65,'n',.64)]:d.feature(G,xx,yy,tx,'vision' if tx!='n' else 'qmod',si)
 label(G,.30,2.85,1.9,.32,'256D',20,color='muted');label(G,.15,6.25,2.1,.35,'196 × 256',20,color='muted');label(G,.30,8.15,1.9,.32,'NSS · 6D',20,color='muted')
 d.block(G,3.0,1.90,1.8,.90,'QA\nMLP + norm',fill='qmod',stroke='qb',size=24);d.feature(G,5.55,2.35,'qA','qmod',.58)
 d.op(G,7.25,2.35,'+',.50);d.block(G,9.0,1.90,1.55,.90,'H_QG',fill='qmod',stroke='qb',head=True,size=26);d.feature(G,11.35,2.35,'sG','qmod',.58)
 d.edge(G,[(1.32,2.35),(3.0,2.35)],name='g_to_QA');d.dot(G,2.05,2.35)
 d.edge(G,[(2.05,2.35),(2.05,1.30),(7.25,1.30),(7.25,2.10)],name='g_residual_skip')
 d.edge(G,[(4.8,2.35),(5.26,2.35)],name='QA_to_qA');d.edge(G,[(5.84,2.35),(7.0,2.35)],name='Scaled_quality_residual')
 label(G,7.70,2.94,3.0,.42,'α·sg(‖g‖)·qA',22);d.edge(G,[(7.50,2.35),(9.0,2.35)],name='Global_residual_to_head');d.edge(G,[(10.55,2.35),(11.06,2.35)],name='Quality_global_score')
 d.op(G,3.15,5.75,'+',.50);d.block(G,4.30,5.28,1.60,.94,'QL\nres. + norm',fill='qmod',stroke='qb',size=22)
 d.pool(G,7.05,5.75);d.feature(G,8.48,5.75,'qL','qmod',.58);d.block(G,9.30,5.28,1.55,.94,'H_QL',fill='qmod',stroke='qb',head=True,size=25);d.feature(G,11.65,5.75,'sL','qmod',.58)
 d.edge(G,[(1.34,5.75),(2.90,5.75)],name='P0_to_NSS_add');d.edge(G,[(3.40,5.75),(4.30,5.75)],name='NSS_augmented_to_QL');d.edge(G,[(5.90,5.75),(6.76,5.75)],name='QL_to_pool');d.edge(G,[(7.34,5.75),(8.19,5.75)],name='Local_quality_representation');d.edge(G,[(8.77,5.75),(9.30,5.75)],name='qL_to_head');d.edge(G,[(10.85,5.75),(11.36,5.75)],name='Local_quality_score')
 d.block(G,2.7,7.18,1.8,.94,'NSS MLP',fill='qmod',stroke='qb',size=24)
 d.edge(G,[(1.32,7.65),(2.7,7.65)],name='n_to_NSS_MLP');d.dot(G,1.95,7.65)
 d.edge(G,[(3.15,7.18),(3.15,6.0)],name='NSS_embedding_to_token_add')
 d.edge(G,[(1.95,7.65),(1.95,8.48),(6.19,8.48)],name='Raw_NSS_to_gate_left',arrow=False);d.edge(G,[(6.31,8.48),(7.05,8.48),(7.05,6.05)],name='Raw_NSS_to_pool_gate')
 label(G,9.15,7.00,2.60,.65,'P: NSS-gated\npooling',21,color='muted')
 d.op(G,13.30,4.03,'M',.60,'qmod');d.feature(G,15.05,4.03,'S_Q','qmod',.72)
 d.edge(G,[(11.64,2.35),(13.3,2.35),(13.3,3.73)],name='Global_score_mix');d.edge(G,[(11.94,5.75),(13.3,5.75),(13.3,4.33)],name='Local_score_mix');d.edge(G,[(13.6,4.03),(14.69,4.03)],name='Quality_scalar_output')
 label(G,10.0,3.60,2.5,.8,'(1−β)sG\n+ βsL',25);label(G,14.08,3.10,1.85,.40,'Quality',25,True)
 # qA and qL are separate from the scoring residual hG.
 d.dot(G,6.25,2.35);d.edge(G,[(6.25,2.35),(6.25,5.69)],name='qA_training_tap1',arrow=False);d.edge(G,[(6.25,5.81),(6.25,9.65),(9.2,9.65)],name='qA_training_tap2')
 d.dot(G,9.0,5.75);d.edge(G,[(9.0,5.75),(9.0,9.59)],name='qL_training_upper',arrow=False);d.edge(G,[(9.0,9.71),(9.0,10.0),(9.2,10.0)],name='qL_training_tap')
 d.shape(G,2.45,9.12,13.45,2.70,'white','border',name='Quality_training_supervision_taps')
 # Reorder this region background under existing tap lines.
 bg=G[-1];bg._element.getparent().remove(bg._element);G._spTree.insert(2,bg._element)
 label(G,2.75,9.23,3.10,.35,'Pretraining taps',23,True)
 d.block(G,9.20,9.40,3.25,.90,'q = norm[(1−ρ)qA + ρqL]',fill='qmod',stroke='qb',size=20);d.edge(G,[(12.45,9.85),(13.0,9.85)],'supervision',name='q_to_component_anchors');d.block(G,13.0,9.40,2.25,.90,'Component anchors\n+ ordinal',fill='q',stroke='qb',size=21)
 d.dot(G,6.25,9.65);d.edge(G,[(6.25,9.65),(6.25,11.05),(9.20,11.05)],name='qA_to_aux_heads')
 d.dot(G,9.0,10.0);d.edge(G,[(9.0,10.0),(9.0,10.99)],name='qL_aux_upper',arrow=False);d.edge(G,[(9.0,11.11),(9.0,11.35),(9.20,11.35)],name='qL_to_aux_heads')
 d.block(G,9.20,10.63,3.25,.84,'C(qA,qL) → H_art / H_aes',fill='qmod',stroke='qb',size=21);d.edge(G,[(12.45,11.05),(13.0,11.05)],'supervision',name='auxiliary_component_targets');d.block(G,13.0,10.63,2.25,.84,'Aux. reg. / rank',fill='q',stroke='qb',size=21)
 label(G,2.75,10.2,3.10,.73,'C = [qA,qL,|qA−qL|]\nClean-view targets',21,color='muted')
 label(G,.45,11.93,15.6,.26,'QAM takes g, P₀ and NSS n. It never routes the Quality scoring path through PI.',20,color='muted')
 anchor_native_connections(p);s.notes_slide.notes_text_frame.text='QAM input g is the global semantic Linear projection, not PI(g). Local P0 is independently projected. NSS n is raw six-dimensional descriptor: embed and add to P0, and feed raw n into gate([local_token,n]). QL is a residual adapter plus normalization. Pool normalizes the pooled local feature; local scalar is predicted after pooling. QA normalized output qA enters the magnitude-matched alpha residual on g. Beta is scalar score mix. q uses rho feature mix and does not replace the score residual. Clean training qA/qL taps supervise fixed component anchors/ordinal and auxiliary heads. Quality text CE remains off. Training tap lines are representation exports, not a third scoring task.'
 p.save(out/'02_QAM_module.pptx')


def aam(out):
 p,s=canvas(16.5,10.0);G=d.group(s.shapes,'AAM_Alignment_Assessment_Module').shapes
 label(G,.4,.28,15.7,.62,'AAM · Alignment Assessment Module',32,True)
 d.shape(G,2.45,1.06,13.45,8.10,'a','ab',name='Alignment_region')
 for yy,lab,fill in [(2.15,'v','vision'),(5.30,'P₀','vision'),(8.30,'t','text')]:d.feature(G,1.0,yy,lab,fill,.70)
 label(G,.10,2.72,2.2,.36,'v = norm PI(g)',20,color='muted');label(G,.10,5.88,2.2,.36,'P₀ bypasses PI',20,color='muted')
 for yy,tx in [(1.68,'AG'),(4.83,'AP'),(7.83,'AT')]:d.block(G,3.0,yy,1.60,.94,tx+'\nadapter',fill='amod',stroke='ab',size=25)
 for yy in [2.15,5.30,8.30]:d.edge(G,[(1.35,yy),(3.0,yy)],name='AAM_input_'+str(yy))
 d.feature(G,5.5,2.15,'aG','amod',.60);d.feature(G,5.5,8.30,'aT','amod',.60)
 d.edge(G,[(4.6,2.15),(5.2,2.15)],name='AG_feature');d.edge(G,[(4.6,8.30),(5.2,8.30)],name='AT_feature')
 d.op(G,7.30,2.15,'C',.50,'amod');d.block(G,9.0,1.68,1.6,.94,'H_AG',fill='amod',stroke='ab',head=True,size=26);d.feature(G,11.35,2.15,'sG','amod',.60)
 d.edge(G,[(5.8,2.15),(7.05,2.15)],name='aG_to_pair');d.edge(G,[(7.55,2.15),(9.0,2.15)],name='Global_pair_to_head');d.edge(G,[(10.6,2.15),(11.05,2.15)],name='Global_alignment_score')
 d.block(G,5.70,4.83,1.85,.94,'W_I / W_T\nText pool',fill='amod',stroke='ab',size=23);d.edge(G,[(4.6,5.30),(5.70,5.30)],name='AP_to_text_pool')
 d.feature(G,8.15,5.30,'e','amod',.55);d.op(G,8.95,5.30,'C',.50,'amod');d.block(G,10.0,4.83,1.50,.94,'H_AP',fill='amod',stroke='ab',head=True,size=25);d.feature(G,12.10,5.30,'sP','amod',.58)
 d.edge(G,[(7.55,5.30),(7.875,5.30)],name='Text_weighted_evidence');d.edge(G,[(8.425,5.30),(8.70,5.30)],name='Evidence_to_pair');d.edge(G,[(9.20,5.30),(10.0,5.30)],name='Local_pair_to_head');d.edge(G,[(11.5,5.30),(11.81,5.30)],name='Local_alignment_score')
 # aT: separate input port to pooling, to both C operators, and to score gate.
 d.edge(G,[(5.80,8.30),(7.75,8.30),(7.75,5.37)],name='Text_branch_lower',arrow=False);d.edge(G,[(7.75,5.23),(7.75,3.10),(7.30,3.10),(7.30,2.40)],name='Text_to_global_pair')
 d.dot(G,7.75,6.15);d.edge(G,[(7.75,6.15),(6.62,6.15),(6.62,5.77)],name='Text_controls_pool');d.edge(G,[(7.75,6.15),(8.95,6.15),(8.95,5.55)],name='Text_to_local_pair')
 d.block(G,13.10,4.02,1.45,1.10,'Gate γ\nM',fill='amod',stroke='ab',size=25);d.feature(G,15.25,4.57,'S_A','amod',.68)
 d.edge(G,[(14.55,4.57),(14.91,4.57)],name='Alignment_scalar_output')
 d.edge(G,[(11.65,2.15),(13.90,2.15),(13.90,4.02)],name='Global_score_to_gate');d.edge(G,[(12.39,5.30),(12.73,5.30),(12.73,4.95),(13.10,4.95)],name='Local_score_to_gate')
 d.dot(G,6.05,2.15);d.edge(G,[(6.05,2.15),(6.05,1.24),(13.40,1.24),(13.40,4.02)],name='Global_feature_to_gate')
 d.dot(G,6.0,8.30);d.edge(G,[(6.0,8.30),(13.40,8.30),(13.40,5.12)],name='Text_feature_to_gate')
 label(G,10.0,6.3,2.4,.9,'γsG +\n(1−γ)sP',25);label(G,14.15,3.25,1.8,.4,'Alignment',23,True)
 label(G,2.80,8.66,12.7,.30,'t = norm PT(T(Eₜ(P)));   C(u,t) = [u,t,|u−t|]',22,color='muted')
 label(G,.45,9.35,15.6,.40,'Training taps: aᶜ = norm(AG(vᶜ));   tP = norm(W_T(aT)). Answer tA comes directly from PT, outside AAM.',20,color='muted')
 anchor_native_connections(p);s.notes_slide.notes_text_frame.text='PI and PT live in the shared representation stem, not in AAM. AAM receives image vector v=normalize PI(g), raw P0 tokens that bypass PI, and text vector t=normalize PT(T(ET(Prompt))). AG/AP/AT are distinct residual adapters. Text pool projects AP(P0) through WI and text aT through WT, normalizes both, computes dot-product softmax over tokens and pools AP(P0). C is [u,t,abs(u-t)]. The adaptive score gate consumes aG,aT,global and local scores. No cross-attention. For SRM image-text, aC=normalize(aG); Prompt tP=normalize WT(aT); Answer tA does not go through AT/WT and bypasses AAM. Matching WI/WT and scoring heads/gate reset downstream while adapters inherit.'
 p.save(out/'03_AAM_module.pptx')
def srm_expansion(out):
 p,s=canvas(18,10.7);G=d.group(s.shapes,'SRM_Symmetric_Stop_Gradient_Module').shapes
 label(G,.45,.32,17.1,.62,'SRM · SimSiam Relation Module',32,True,name='SRM_title')
 label(G,.65,1.0,16.7,.45,'Feature inputs from the shared stem / AAM · predictors are reused by modality',23,color='muted',name='SRM_scope')
 d.shape(G,3.12,1.75,11.68,5.68,'proj','border',name='SRM_boundary')
 d.feature(G,1.10,4.60,'u','vision',.80);d.feature(G,16.90,4.60,'v','text',.80)
 d.edge(G,[(1.50,4.60),(2.28,4.60)],name='u_input');d.dot(G,2.28,4.60)
 d.edge(G,[(16.50,4.60),(15.72,4.60)],name='v_input');d.dot(G,15.72,4.60)
 d.block(G,3.75,2.12,2.15,1.10,'Predictor hₘ\n+ norm',fill='vision',size=25)
 d.block(G,12.10,5.80,2.15,1.10,'Predictor hₙ\n+ norm',fill='text',stroke='tb',size=25)
 d.block(G,3.75,5.80,2.15,1.10,'sg(u)',fill='white',stroke='train',size=27)
 d.block(G,12.10,2.12,2.15,1.10,'sg(v)',fill='white',stroke='train',size=27)
 d.edge(G,[(2.28,4.60),(2.28,2.67),(3.75,2.67)],name='u_to_predictor')
 d.edge(G,[(2.28,4.60),(2.28,6.35),(3.75,6.35)],name='u_to_sg')
 d.edge(G,[(15.72,4.60),(15.72,2.67),(14.25,2.67)],name='v_to_sg')
 d.edge(G,[(15.72,4.60),(15.72,6.35),(14.25,6.35)],name='v_to_predictor')
 d.block(G,7.08,2.06,3.84,1.22,'D(hₘ(u), sg(v))\nnegative cosine',fill='white',stroke='feat',size=25)
 d.block(G,7.08,5.74,3.84,1.22,'D(hₙ(v), sg(u))\nnegative cosine',fill='white',stroke='feat',size=25)
 d.edge(G,[(5.90,2.67),(7.08,2.67)],name='Pred_u_to_D')
 d.edge(G,[(12.10,2.67),(10.92,2.67)],name='sg_v_to_D')
 d.edge(G,[(5.90,6.35),(7.08,6.35)],name='sg_u_to_D')
 d.edge(G,[(12.10,6.35),(10.92,6.35)],name='Pred_v_to_D')
 d.block(G,7.80,4.16,2.40,.85,'½ × sum',fill='white',stroke='feat',size=29)
 d.edge(G,[(9.00,3.28),(9.00,4.16)],name='D1_to_average')
 d.edge(G,[(9.00,5.74),(9.00,5.01)],name='D2_to_average')
 d.edge(G,[(10.20,4.585),(12.05,4.585)],'supervision',name='SRM_loss_output')
 d.block(G,12.05,4.16,2.15,.85,'L_SRM',fill='white',stroke='train',size=29)
 # Explicit call signatures remove any ambiguity about four independent networks.
 label(G,.60,7.77,16.8,.46,'Four primary relations · two shared modality-specific predictor parameter sets',23,True,name='Predictor_sharing_rule')
 rows=[('SRM 1 · I–I','v₁ , v₂','h_I / h_I'),('SRM 2 · I–Prompt','aᶜ , tP','h_I / h_T'),('SRM 3 · I–Answer','aᶜ , tA','h_I / h_T'),('SRM 4 · Prompt–Answer','tP , tA','h_T / h_T')]
 for j,(call,ports,preds) in enumerate(rows):
  x=.58+j*4.27
  d.shape(G,x,8.46,4.02,1.63,'white','border',name='SRM_call_'+str(j+1))
  label(G,x+.1,8.58,3.82,.35,call,22,True)
  label(G,x+.1,9.08,3.82,.34,'Inputs: '+ports,22)
  label(G,x+.1,9.55,3.82,.34,'Predictors: '+preds,21,color='muted')
 label(G,.60,10.20,16.8,.31,'PI/PT remain in the shared stem and transfer downstream; h_I/h_T are pretraining-only predictors (256 → 128 → 256).',20,color='muted',name='Projector_predictor_boundary')
 s.notes_slide.notes_text_frame.text='D(p,t)=-cosine(normalize(p),stopgrad(normalize(t))). Two directed terms are averaged with per-sample weights as configured by the caller. hI and hT are different networks. Repeated hI or hT labels mean shared weights, not independent predictors for every loss. I-I receives PI(g) without AG; image-text relations receive norm(AG(norm(PI(g_clean)))). Text Prompt tP and Answer tA definitions are in AAM expansion. The four calls are four loss terms in one pretraining run, not four pretraining experiments. Additional positive local/word relation calls use the same predictor families but are summarized outside the main four calls. No negative push target or quality text-bank CE is introduced.'
 anchor_native_connections(p)
 p.save(out/'04_SRM_module.pptx')



STEMS=['01_split_framework','02_QAM_module','03_AAM_module','04_SRM_module']
def render(out):
 exe=shutil.which('soffice') or shutil.which('libreoffice');report=[];combined=fitz.open()
 for stem in STEMS:
  ppt=out/(stem+'.pptx');subprocess.run([exe,'--headless','--convert-to','pdf','--outdir',str(out),str(ppt)],check=True,timeout=120)
  pdf=fitz.open(out/(stem+'.pdf'));page=pdf[0];scale=5600/max(page.rect.width,page.rect.height);pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False);pix.save(out/(stem+'.png'))
  page.get_pixmap(matrix=fitz.Matrix(1700/page.rect.width,1700/page.rect.width),alpha=False).save(out/(stem+'_preview.png'))
  assert not page.get_images() and '\ufffd' not in page.get_text()
  spans=[x for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for x in l['spans']];assert all(page.rect.contains(fitz.Rect(x['bbox'])) for x in spans)
  text=page.get_text();(out/(stem+'_text.txt')).write_text(text);assert 'DAC' not in text,(stem,'obsolete DAC label')
  combined.insert_pdf(pdf);report.append(dict(name=stem,pixels=[pix.width,pix.height],vector_pdf=True,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ppt,out/(stem+'.pdf'),out/(stem+'.png')]}))
 combined.save(out/'all_split_figures.pdf');(out/'render_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 overview(out);qam(out);aam(out);srm_expansion(out);render(out)
