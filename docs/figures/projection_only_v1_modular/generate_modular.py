#!/usr/bin/env python3
"""Modular editable figures: overview, DAC expansion, SRM expansion.
Run python generate_modular.py --output-dir rendered (LibreOffice required).
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,subprocess,shutil,hashlib,zipfile
import fitz
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import drawing_primitives as d

HERE=Path(__file__).resolve().parent

def canvas(w,h):
 p=Presentation();p.slide_width=Inches(w);p.slide_height=Inches(h)
 s=p.slides.add_slide(p.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white')
 return p,s

def label(sh,x,y,w,h,text,size=24,bold=False,color='ink',name='label'):
 return d.txt(sh,x,y,w,h,text,size,bold,color,PP_ALIGN.CENTER,name)

def dac_icon(sh,x,y,w,h,note,stage):
 g=d.group(sh,'DAC_'+stage)
 d.shape(g.shapes,x,y,w,h,'white','feat',name='DAC_body_'+stage,width=1.8)
 label(g.shapes,x+.15,y+.32,w-.3,.62,'DAC',38,True,name='DAC_ID_'+stage)
 label(g.shapes,x+.20,y+1.01,w-.4,1.0,'Dual-Branch\nAssessment Core',27,True,name='DAC_name_'+stage)
 # A deliberate schematic icon, expanded separately; not a second computation graph.
 d.block(g.shapes,x+.38,y+2.28,w-.76,.80,'Visual + text encoding',fill='vision',size=21)
 d.block(g.shapes,x+.38,y+3.31,w-.76,.73,'Global / local projections',fill='proj',size=21)
 d.block(g.shapes,x+.38,y+4.27,(w-.94)/2,1.03,'Quality',fill='qmod',stroke='qb',size=22)
 d.block(g.shapes,x+.56+(w-.94)/2,y+4.27,(w-.94)/2,1.03,'Alignment',fill='amod',stroke='ab',size=21)
 label(g.shapes,x+.18,y+5.72,w-.36,1.24,note,21,color='muted',name='DAC_execution_policy_'+stage)
 return g

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


def overview(out):
 p,s=canvas(24,13.10);A=d.group(s.shapes,'A_Pretraining_Overview').shapes;B=d.group(s.shapes,'B_Downstream_Overview').shapes;L=d.group(s.shapes,'Overview_Legend').shapes
 d.shape(A,.22,.22,15.30,10.00,'white','border',name='Pretraining_panel')
 d.shape(B,15.86,.22,7.92,10.00,'white','border',name='Downstream_panel')
 d.txt(A,.55,.49,13.8,.6,'(a) Pretraining',33,True,align=PP_ALIGN.LEFT,name='Pretraining_title')
 d.txt(B,16.12,.49,7.25,.6,'(b) Downstream assessment',30,True,align=PP_ALIGN.LEFT,name='Downstream_title')
 label(A,.65,1.31,6.10,.40,'Three image views · one set of core weights',22,color='muted',name='Shared_core_views')
 label(A,9.94,1.31,5.17,.40,'Four primary SRM relations',22,color='muted',name='Four_relations')
 dac_icon(A,3.20,1.95,3.8,7.40,'V/G/PI reused across views\nScoring heads: Clean only\nExpanded in DAC figure','pretrain')
 for y,text in ((2.26,'SafeAug I₁'),(3.36,'SafeAug I₂'),(4.76,'Clean Iᶜ')):
  d.image_tag(A,.67,y,text,1.98,.68)
  d.edge(A,[(2.65,y+.34),(3.20,y+.34)],name='Pretrain_'+text)
 for y,text in ((6.22,'Prompt P'),(7.62,'Answer A')):
  d.block(A,.67,y,1.98,.76,text,fill='text',stroke='tb',size=24)
  d.edge(A,[(2.65,y+.38),(3.20,y+.38)],name='Pretrain_'+text)
 label(A,.65,8.71,2.02,.69,'5 trainable\ncontext tokens',21,color='muted',name='Five_context_tokens')
 # Each feature is emitted once. Repeated uses are literal solid-line forks.
 for yy,lab,fill in ((2.56,'v₁','vision'),(3.18,'v₂','vision'),(4.35,'aᶜ','vision'),(5.02,'tP','text'),(6.45,'tA','text')):
  d.edge(A,[(7.00,yy),(7.37,yy)],name='DAC_feature_'+lab)
  d.feature(A,7.62,yy,lab,fill,.50)
 d.edge(A,[(7.87,2.56),(10.25,2.56)],name='v1_to_II')
 d.edge(A,[(7.87,3.18),(10.25,3.18)],name='v2_to_II')
 d.edge(A,[(7.87,4.35),(10.25,4.35)],name='aC_to_IP')
 d.dot(A,8.30,4.35)
 d.edge(A,[(8.30,4.35),(8.30,4.96)],name='aC_reuse_upper',arrow=False)
 d.edge(A,[(8.30,5.08),(8.30,6.05),(10.25,6.05)],name='aC_to_IA')
 d.edge(A,[(7.87,5.02),(8.82,5.02),(8.82,4.85),(10.25,4.85)],name='tP_to_IP')
 d.dot(A,8.82,5.02)
 d.edge(A,[(8.82,5.02),(8.82,5.99)],name='tP_reuse_upper',arrow=False)
 d.edge(A,[(8.82,6.11),(8.82,6.39)],name='tP_reuse_middle',arrow=False)
 d.edge(A,[(8.82,6.51),(8.82,7.45),(10.25,7.45)],name='tP_to_PA')
 d.edge(A,[(7.87,6.45),(10.25,6.45)],name='tA_to_IA')
 d.dot(A,9.32,6.45)
 d.edge(A,[(9.32,6.45),(9.32,7.39)],name='tA_reuse_upper',arrow=False)
 d.edge(A,[(9.32,7.51),(9.32,8.02),(10.25,8.02)],name='tA_to_PA')
 for i,(y,title,loss) in enumerate(((2.24,'I–I','L_II'),(4.02,'I–Prompt','L_IP'),(5.72,'I–Answer','L_IA'),(7.15,'Prompt–Answer','L_PA')),1):
  d.block(A,10.25,y,2.60,1.20,f'SRM {i}\n{title}',fill='proj',stroke='feat',size=25)
  d.edge(A,[(12.85,y+.60),(13.30,y+.60)],'supervision',name=loss)
  d.block(A,13.30,y+.26,1.50,.68,loss,fill='white',stroke='train',size=24)
 # Human losses are intentionally outside the four relation calls.
 d.block(A,10.25,8.82,4.55,.91,'Clean-view supervised objectives\nQuality / Align · components / ordinal',fill='q',stroke='qb',size=21)
 d.edge(A,[(7.0,8.98),(10.25,8.98)],'supervision',name='Clean_supervision_exports')
 label(A,7.07,8.42,2.96,.44,'S_Q, S_A, q, aux.',21,name='Clean_supervision_ports')
 label(A,7.12,9.84,7.79,.32,'PCGrad · positive local / word supervision retained.',20,color='muted',name='Local_word_objectives')
 # Identical visual abstraction in downstream; weights initialized, then independent.
 dac_icon(B,18.25,1.95,3.8,7.40,'One image + prompt\nInherited representations\nNew dataset scoring heads','downstream')
 d.image_tag(B,16.15,3.0,'Image I',1.72,.74)
 d.block(B,16.15,6.35,1.72,.80,'Prompt P',fill='text',stroke='tb',size=24)
 d.edge(B,[(17.87,3.37),(18.25,3.37)],name='Downstream_image')
 d.edge(B,[(17.87,6.75),(18.25,6.75)],name='Downstream_prompt')
 for yy,lab,fill in ((4.48,'Quality\nS_Q','qmod'),(7.10,'Align\nS_A','amod')):
  d.edge(B,[(22.05,yy),(22.43,yy)],name='Downstream_'+lab)
  d.block(B,22.43,yy-.62,1.22,1.24,lab,fill=fill,size=23)
 label(B,16.18,9.63,7.3,.35,'Two task outputs · no SimSiam loss at inference',20,color='muted',name='Two_outputs')
 # One transfer bundle, with no training head inheritance implied.
 d.block(L,4.27,10.62,1.82,.62,'θrepr (pre)',fill='text',stroke='transfer',size=24)
 d.block(L,19.17,10.62,1.82,.62,'θrepr (init)',fill='text',stroke='transfer',size=24)
 d.edge(L,[(6.09,10.93),(19.17,10.93)],'transfer',name='Representation_initialization_only')
 label(L,7.02,10.39,11.40,.40,'Transfer representations; reset dataset heads, matching Linear layers and score gates',21,color='transfer',name='Transfer_scope')
 label(L,.55,11.47,22.90,.44,'DAC is reused as an architectural module. Its weights are shared across calls within a stage, not tied across stages.',22,color='muted',name='Architecture_vs_weight_sharing')
 legend(L,12.12)
 anchor_native_connections(p)
 p.save(out/'01_modular_framework.pptx')
 s.notes_slide.notes_text_frame.text='DAC = Dual-Branch Assessment Core, SRM = SimSiam Relation Module. DAC is a drawing abstraction over existing functions, not a new implementation. View1/View2 call image feature paths; Clean View executes scoring branches. Four SRM calls correspond to image-image, image-prompt, image-answer, prompt-answer losses. Predictors are reused by modality: hI for image inputs and hT for text inputs. Features v1,v2,aC,tP,tA are defined in DAC expansion. Clean labels and local/word objectives are additional retained losses, not replaced by the four SimSiam terms. The theta transfer excludes dataset heads, alignment matching Linear layers and score gates. Frozen text base weights are reloaded from the same CLIP source; five learned context tokens transfer.'
 p.save(out/'01_modular_framework.pptx')


def legend(sh,y):
 d.shape(sh,.23,y,23.54,.72,'white','border',name='Line_semantics')
 d.edge(sh,[(.60,y+.35),(1.12,y+.35)],name='Legend_data')
 label(sh,1.20,y+.14,2.3,.41,'Forward features',20)
 d.edge(sh,[(3.73,y+.35),(4.25,y+.35)],'supervision',name='Legend_loss')
 label(sh,4.33,y+.14,3.2,.41,'Training supervision',20)
 d.edge(sh,[(7.82,y+.35),(8.34,y+.35)],'transfer',name='Legend_transfer')
 label(sh,8.42,y+.14,3.1,.41,'Parameter initialization',20)
 d.dot(sh,12.00,y+.35,.055);label(sh,12.17,y+.14,2.40,.41,'Feature reuse',20)
 label(sh,14.97,y+.14,8.11,.41,'SRM 1–4: same computation; shared h_I / h_T by modality',20)


def dac_expansion(out,source):
 src=Presentation(source);original=next(s for s in src.slides[0].shapes if s.name=='Panel_B_Downstream_Assessment')
 p,s=canvas(14.55,15.60)
 element=deepcopy(original._element);s.shapes._spTree.insert_element_before(element,'p:extLst')
 g=s.shapes[-1];g.left=Inches(.35);g.top=Inches(.25)
 def walk(shapes):
  for z in shapes:
   if z.shape_type==6:yield from walk(z.shapes)
   else:yield z
 for z in walk(g.shapes):
  if z.name=='Downstream_title':
   tf=z.text_frame;tf.paragraphs[0].runs[0].text='DAC · Dual-Branch Assessment Core'
   for r in tf.paragraphs[0].runs:r.font.size=Pt(27)
  if z.name=='Downstream_reset_note':
   z.text_frame.paragraphs[0].runs[0].text='Image-only Quality · text-conditioned Alignment'
 L=d.group(s.shapes,'DAC_Exported_Feature_Ports').shapes
 d.shape(L,.35,13.24,13.82,2.08,'white','border',name='Feature_port_definitions')
 d.txt(L,.65,13.43,13.1,.39,'Training feature taps exported by DAC',25,True,align=PP_ALIGN.LEFT,name='Export_ports_title')
 d.txt(L,.65,13.98,13.1,.40,'v = norm(PI(g));   aᶜ = norm(AG(vᶜ));   q = norm[(1−ρ)qA + ρqL]',22,align=PP_ALIGN.LEFT,name='Image_quality_feature_ports')
 d.txt(L,.65,14.47,13.1,.40,'tP = norm(Wₜ(AT(norm(PT(T(Eₜ(P)))))));   tA = norm(PT(T(Eₜ(A))))',21,align=PP_ALIGN.LEFT,name='Exact_text_feature_ports')
 d.txt(L,.65,14.93,13.1,.27,'Frozen Eₜ base weights; 5 trainable context tokens. PI/PT are feature projectors, not scoring heads.',18,align=PP_ALIGN.LEFT,color='muted',name='DAC_scope_note')
 s.notes_slide.notes_text_frame.text='Expanded DAC reuses the source-verified native editable downstream diagram. G is semantic_proj, L contains the two 1x1 local projections; no global-local cross-attention. Quality uses g->QA residual and independent NSS-local path. P0 bypasses PI. Alignment combines global matching and text-guided pooled local evidence. aC differs from scoring aG by final L2 normalization for semantic SimSiam relations. tP includes AT and Wt; tA does not. v1/v2 are obtained separately from two augmented images using shared weights. Predictor heads are NOT inside DAC; they live in SRM. Same architectural module is used in pretraining and downstream, with representation initialization only across stages.'
 p.save(out/'02_DAC_module.pptx')


def srm_expansion(out):
 p,s=canvas(18,10.7);G=d.group(s.shapes,'SRM_Symmetric_Stop_Gradient_Module').shapes
 label(G,.45,.32,17.1,.62,'SRM · SimSiam Relation Module',32,True,name='SRM_title')
 label(G,.65,1.0,16.7,.45,'Projected feature inputs from DAC · predictors are reused by modality',23,color='muted',name='SRM_scope')
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
 label(G,.60,10.20,16.8,.31,'PI/PT remain inside DAC and transfer downstream; h_I/h_T are pretraining-only predictors (256 → 128 → 256).',20,color='muted',name='Projector_predictor_boundary')
 s.notes_slide.notes_text_frame.text='D(p,t)=-cosine(normalize(p),stopgrad(normalize(t))). Two directed terms are averaged with per-sample weights as configured by the caller. hI and hT are different networks. Repeated hI or hT labels mean shared weights, not independent predictors for every loss. I-I receives PI(g) without AG; image-text relations receive norm(AG(norm(PI(g_clean)))). Text Prompt tP and Answer tA definitions are in DAC expansion. The four calls are four loss terms in one pretraining run, not four pretraining experiments. Additional positive local/word relation calls use the same predictor families but are summarized outside the main four calls. No negative push target or quality text-bank CE is introduced.'
 anchor_native_connections(p)
 p.save(out/'03_SRM_module.pptx')


def render_all(out):
 exe=shutil.which('soffice') or shutil.which('libreoffice')
 report=[];combined=fitz.open()
 for stem in ['01_modular_framework','02_DAC_module','03_SRM_module']:
  ppt=out/(stem+'.pptx')
  subprocess.run([exe,'--headless','--convert-to','pdf','--outdir',str(out),str(ppt)],check=True,timeout=120)
  pdf=fitz.open(out/(stem+'.pdf'));page=pdf[0];pix=page.get_pixmap(matrix=fitz.Matrix(5600/page.rect.width,5600/page.rect.width),alpha=False);pix.save(out/(stem+'.png'))
  page.get_pixmap(matrix=fitz.Matrix(1700/page.rect.width,1700/page.rect.width),alpha=False).save(out/(stem+'_preview.png'))
  text=page.get_text();assert '\ufffd' not in text
  assert not page.get_images()
  shapes={'native_shapes':0,'connectors':0,'text_objects':0,'pictures':0}
  def count(items):
   for z in items:
    shapes['native_shapes']+=1
    if z.shape_type==6:count(z.shapes)
    if z.shape_type==9:shapes['connectors']+=1
    if z.shape_type==13:shapes['pictures']+=1
    if z.has_text_frame and z.text:shapes['text_objects']+=1
  count(Presentation(ppt).slides[0].shapes)
  spans=[q for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for q in l['spans']]
  outside=[q['text'] for q in spans if not page.rect.contains(fitz.Rect(q['bbox']))]
  assert not outside,(stem,outside)
  (out/(stem+'_text.txt')).write_text(text)
  combined.insert_pdf(pdf)
  report.append(dict(name=stem,pixels=[pix.width,pix.height],vector_pdf=True,objects=shapes,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ppt,out/(stem+'.pdf'),out/(stem+'.png')]}))
 combined.save(out/'all_modular_figures.pdf')
 (out/'render_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);ap.add_argument('--source-pptx',type=Path,default=HERE/'assets/source_DAC_native.pptx')
 a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(exist_ok=True,parents=True)
 overview(out);dac_expansion(out,a.source_pptx);srm_expansion(out);render_all(out)
