#!/usr/bin/env python3
"""Publication SRM diagram. All geometry and math labels are native PowerPoint objects.
Run: python generate_srm.py --output-dir ./rendered
"""
from pathlib import Path
import argparse,subprocess,shutil,json,hashlib
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE,MSO_SHAPE_TYPE
import fitz
import drawing_primitives as d
from PIL import ImageFont
HERE=Path(__file__).resolve().parent


def math(sh,x,y,w,h,string,size=27,name='math',color='ink'):
 # Native positioned glyph runs keep sub/superscripts consistent in PPT and PDF.
 # Office/LibreOffice apply different implicit font reductions to baseline attrs;
 # explicit positions avoid that discrepancy, while every glyph stays editable.
 group=d.group(sh,name);parts=[];i=0
 while i<len(string):
  if string[i] in '_^' and i+1<len(string) and string[i+1]=='{':
   j=string.index('}',i+2);value=string[i+2:j]
   parts.append((value,.67,.27 if string[i]=='_' else -.38,value not in ('SRM','I','T') and not value.isdigit()));i=j+1;continue
  f=next((f for f in ['sg','cos','norm'] if string.startswith(f,i)),None)
  if f:parts.append((f,1,0,False));i+=len(f);continue
  char=string[i];parts.append((char,1,0,char.isalpha() and char!='Σ'));i+=1
 def font(fs,italic):
  fn='DejaVuSerif'+('-Italic' if italic else '')+'.ttf'
  p=HERE/'fonts'/fn
  if not p.exists():p=Path('/usr/share/fonts/truetype/dejavu')/fn
  return ImageFont.truetype(str(p),round(fs*8))
 widths=[font(size*factor,italic).getlength(text)/8 for text,factor,shift,italic in parts]
 width=sum(widths)
 if width>(w*72-2):
  ratio=(w*72-2)/width;size*=ratio;widths=[v*ratio for v in widths];width=sum(widths)
 xx=x+(w-width/72)/2
 y0=y+(h-size*1.40/72)/2
 for j,((text,factor,shift,italic),advance) in enumerate(zip(parts,widths)):
  fs=size*factor
  top=y0+(size*.95+shift*size-fs*.95)/72
  o=d.nm(group.shapes.add_textbox(Inches(xx),Inches(top),Inches(max(advance/72+.015,.03)),Inches(fs*1.45/72)),name+'_'+str(j))
  tf=o.text_frame;tf.clear();tf.word_wrap=False;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.TOP
  p=tf.paragraphs[0];p.alignment=PP_ALIGN.LEFT;p.space_before=p.space_after=Pt(0)
  r=p.add_run();r.text=text;r.font.name='DejaVu Serif';r.font.size=Pt(fs);r.font.italic=italic;r.font.color.rgb=d.rgb(color)
  xx+=advance/72
 return group


def feature(sh,x,y,letter):
 d.shape(sh,x-.35,y-.35,.70,.70,'vision','feat',MSO_SHAPE.OVAL,'Feature_'+letter,width=1.3)
 math(sh,x-.35,y-.35,.70,.70,letter,30,'Feature_label_'+letter)


def predictor(sh,x,y,suffix):
 G=d.group(sh,'Predictor_h_'+suffix).shapes
 d.shape(G,x,y,2.0,1.1,'DDF1ED','299081',name='Predictor_body_'+suffix,width=1.2)
 d.txt(G,x+.08,y+.08,1.84,.36,'Predictor',20,True,name='Predictor_caption_'+suffix)
 math(G,x+.08,y+.48,1.84,.48,'h_{'+suffix+'}',29,'Predictor_symbol_'+suffix)


def sg(sh,x,y,arg):
 d.shape(sh,x,y,2.0,1.1,'FCEADE','CE9470',name='Stop_gradient_body_'+arg,width=1.2)
 math(sh,x+.08,y+.15,1.84,.80,'sg('+arg+')',28,'Stop_gradient_'+arg)


def norm(sh,x,y,name):
 d.shape(sh,x,y,.64,.64,'white','border',name=name,width=1.1)
 math(sh,x,y,.64,.64,'ℓ_{2}',23,name+'_label')


def attach(p):
 items=[]
 def walk(shapes):
  for z in shapes:
   if z.shape_type==6:walk(z.shapes)
   else:items.append(z)
 walk(p.slides[0].shapes)
 targets=[z for z in items if z.shape_type==MSO_SHAPE_TYPE.AUTO_SHAPE and .3<z.width/914400<6 and .3<z.height/914400<4]
 for z in items:
  if z.shape_type!=MSO_SHAPE_TYPE.LINE:continue
  for connect,px,py in [(z.begin_connect,z.begin_x,z.begin_y),(z.end_connect,z.end_x,z.end_y)]:
   found=False
   for t in targets:
    x,y,w,h=t.left,t.top,t.width,t.height
    for idx,(xx,yy) in enumerate([(x+w/2,y),(x,y+h/2),(x+w/2,y+h),(x+w,y+h/2)]):
     if abs(px-xx)+abs(py-yy)<4000:connect(t,idx);found=True;break
    if found:break


def build(out):
 p=Presentation();p.slide_width=Inches(16);p.slide_height=Inches(10.0);s=p.slides.add_slide(p.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white')
 G=d.group(s.shapes,'SRM_Publication_Figure').shapes
 d.shape(G,.42,.30,.10,.65,'299081',None,name='Title_accent'); d.txt(G,.72,.27,13.6,.60,'SimSiam Relation Module',32,True,align=PP_ALIGN.LEFT,name='Figure_title'); d.shape(G,14.15,.31,1.38,.52,'253646',None,name='SRM_badge'); d.txt(G,14.15,.31,1.38,.52,'SRM',22,True,color='white',name='SRM_badge_label')
 d.txt(G,.4,.80,15.2,.36,'Symmetric prediction • Stop-gradient targets • Shared modality predictors',21,color='muted',name='Figure_scope')
 d.shape(G,1.55,1.30,12.9,4.86,'F8FAFC','border',name='SRM_boundary',width=1.1)
 d.shape(G,2.65,1.49,10.70,1.57,'FFFFFF',None,name='Upper_comparison_lane')
 d.shape(G,2.65,4.43,10.70,1.57,'FFFFFF',None,name='Lower_comparison_lane')
 d.txt(G,3.10,1.34,2.0,.27,'PREDICT u',13,True,color='299081',name='Upper_lane_caption')
 d.txt(G,10.90,4.41,2.0,.20,'PREDICT v',12,True,color='299081',name='Lower_lane_caption')
 d.txt(G,3.15,3.48,3.0,.42,'256-D projected features',17,color='muted',name='Input_scope')
 # Two source features, each split into a live prediction branch and a detached target branch.
 feature(G,.75,3.725,'u');feature(G,15.25,3.725,'v')
 d.edge(G,[(1.10,3.725),(2.0,3.725)],name='Input_u');d.dot(G,2.0,3.725,.042)
 d.edge(G,[(14.90,3.725),(14.0,3.725)],name='Input_v');d.dot(G,14.0,3.725,.042)
 predictor(G,3.10,1.70,'m');predictor(G,10.90,4.65,'n')
 sg(G,10.90,1.70,'v');sg(G,3.10,4.65,'u')
 d.edge(G,[(2.0,3.725),(2.0,2.25),(3.10,2.25)],name='u_live_branch')
 d.edge(G,[(2.0,3.725),(2.0,5.20),(3.10,5.20)],name='u_target_branch')
 d.edge(G,[(14.0,3.725),(14.0,2.25),(12.90,2.25)],name='v_target_branch')
 d.edge(G,[(14.0,3.725),(14.0,5.20),(12.90,5.20)],name='v_live_branch')
 norm(G,5.39,1.93,'Normalize_p_u');norm(G,9.97,4.88,'Normalize_p_v')
 for pts,name in [([(5.10,2.25),(5.39,2.25)],'Predict_u_to_norm'), ([(6.03,2.25),(6.65,2.25)],'p_u_to_D'), ([(10.90,5.20),(10.61,5.20)],'Predict_v_to_norm'), ([(9.97,5.20),(9.35,5.20)],'p_v_to_D')]:d.edge(G,pts,name=name)
 math(G,6.00,1.72,.65,.34,'p_{u}',20,'Prediction_p_u')
 math(G,9.35,4.66,.65,.34,'p_{v}',20,'Prediction_p_v')
 for y,formula,name in [(1.70,'D(p_{u},sg(v))','D_u_to_v'),(4.65,'D(p_{v},sg(u))','D_v_to_u')]:
  d.shape(G,6.65,y,2.70,1.10,'EDE9F7','8B79B1',name=name,width=1.3)
  math(G,6.72,y+.12,2.56,.84,formula,25,name+'_math')
 d.edge(G,[(10.90,2.25),(9.35,2.25)],name='Detached_v_to_D')
 d.edge(G,[(5.10,5.20),(6.65,5.20)],name='Detached_u_to_D')
 d.shape(G,7.30,3.325,1.40,.80,'EDE9F7','8B79B1',name='Symmetric_average',width=1.3)
 math(G,7.30,3.325,1.40,.80,'½Σ',29,'Average_symbol')
 d.edge(G,[(8.0,2.80),(8.0,3.325)],name='Upper_term_to_average')
 d.edge(G,[(8.0,4.65),(8.0,4.125)],name='Lower_term_to_average')
 d.edge(G,[(8.70,3.725),(10.98,3.725)],'flow',name='Relation_loss_output')
 math(G,11.05,3.34,1.80,.78,'L_{SRM}',29,'Relation_loss_symbol')
 d.txt(G,10.97,4.06,2.03,.29,'relation loss',17,color='muted',name='Relation_loss_caption')
 # Compact definition band and four reusable call cards.
 d.shape(G,.42,6.37,15.16,.64,'F1F4F8',None,name='Equation_band')
 math(G,.60,6.42,9.3,.48,'L_{SRM}=½[D(p_{u},sg(v))+D(p_{v},sg(u))]',24,'Full_SRM_equation')
 math(G,10.05,6.42,5.2,.48,'D(p,t)=−cos(p,t)',23,'Distance_definition')
 d.txt(G,.46,7.20,8.8,.32,'FOUR RELATIONS · ONE REUSABLE MODULE',18,True,align=PP_ALIGN.LEFT,name='Calls_heading')
 d.txt(G,10.15,7.20,5.35,.32,'Positive matching only in the current mode',15,color='muted',align=PP_ALIGN.RIGHT,name='Current_mode')
 T=d.group(s.shapes,'Four_SRM_Call_Signatures').shapes
 calls=[('I–I','(v_{1},v_{2})','h_{I},h_{I}','1','DDF1ED','299081'),('I–Prompt','(a^{c},t_{P})','h_{I},h_{T}','a','E5EFF8','528CB3'),('I–Answer','(a^{c},t_{A})','h_{I},h_{T}','c','FCEADE','CE9470'),('Prompt–Answer','(t_{P},t_{A})','h_{T},h_{T}','ac','EDE9F7','8B79B1')]
 for i,(title,pair,preds,weight,fill,border) in enumerate(calls):
  x=.42+3.84*i
  C=d.group(T,'Call_card_'+str(i+1)).shapes
  d.shape(C,x,7.72,3.64,1.40,'FFFFFF','border',name='Card_boundary_'+str(i),width=.8)
  d.shape(C,x,7.72,3.64,.39,fill,None,name='Card_header_'+str(i))
  d.txt(C,x+.08,7.76,3.48,.28,title,18,True,color=border,name='Call_title_'+str(i))
  math(C,x+.08,8.17,3.48,.33,'(u,v)='+pair,20,'Call_features_'+str(i))
  math(C,x+.08,8.53,2.32,.32,'('+preds+')',19,'Call_predictors_'+str(i))
  math(C,x+2.36,8.53,1.20,.32,'w='+weight,19,'Call_weight_'+str(i))
 d.txt(G,.45,9.30,15.1,.28,'Two predictor networks are reused by modality across all four calls. PI/PT remain outside SRM.',16,color='muted',name='Sharing_and_module_boundary')
 d.txt(G,.45,9.65,15.1,.25,'sg: stop-gradient     ℓ₂: normalization     Predictor: 256 → 128 → 256     a: human alignment     c: CLIP-derived answer weight',15,color='muted',name='Legend')
 attach(p)
 s.notes_slide.notes_text_frame.text='Source-verified projection_only_v1 SRM. Inputs u,v are already-projected normalized 256D features from the shared stem or AAM taps. h_m/h_n are selected from exactly two predictor networks hI and hT, not independent per-loss predictors. I-I v1/v2 use hI/hI; I-Prompt and I-Answer use hI/hT; Prompt-Answer uses hT/hT. Predictors are Linear256->128, BN, ReLU, Linear128->256, followed by normalization. Target branch stop-gradient does not mean encoder parameters are permanently frozen. D is negative cosine. The displayed equation is per pair; configured per-sample confidence/mask weighting and batch reduction are applied to yield the scalar relation loss. PI/PT are persistent projectors in the shared stem, not inside SRM. Features: v1,v2=norm PI(g1/g2); a^c=norm AG(norm PI(g_clean)); tP=norm WT(AT(norm PT(T(ET(P))))); tA=norm PT(T(ET(A))). Additional local/word positive relation losses remain outside the four primary global/view calls. No quality text-bank CE or negative push introduced.'
 ppt=out/'SRM_publication.pptx';p.save(ppt);return ppt


def render(ppt,out):
 exe=shutil.which('soffice') or shutil.which('libreoffice')
 subprocess.run([exe,'--headless','--convert-to','pdf','--outdir',str(out),str(ppt)],check=True,timeout=120)
 pdf=fitz.open(out/'SRM_publication.pdf');page=pdf[0];pix=page.get_pixmap(matrix=fitz.Matrix(6000/page.rect.width,6000/page.rect.width),alpha=False);pix.save(out/'SRM_publication.png')
 page.get_pixmap(matrix=fitz.Matrix(1800/page.rect.width,1800/page.rect.width),alpha=False).save(out/'preview.png')
 text=page.get_text();assert '\ufffd' not in text and not page.get_images()
 spans=[z for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for z in l['spans']]
 assert all(page.rect.contains(fitz.Rect(z['bbox'])) for z in spans)
 counts=dict(shapes=0,connectors=0,text_objects=0,pictures=0)
 def walk(shapes):
  for x in shapes:
   counts['shapes']+=1
   if x.shape_type==6:walk(x.shapes)
   if x.shape_type==9:counts['connectors']+=1
   if x.shape_type==13:counts['pictures']+=1
   if x.has_text_frame and x.text:counts['text_objects']+=1
 walk(Presentation(ppt).slides[0].shapes)
 report=dict(pixels=[pix.width,pix.height],canvas_inches=[16,10.0],editable_objects=counts,pdf_raster_images=len(page.get_images()),out_of_page_text=0,fonts=sorted(set(f[3] for f in page.get_fonts())),source='Native editable PPTX -> LibreOffice PDF -> PNG',sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ppt,out/'SRM_publication.pdf',out/'SRM_publication.png']})
 (out/'render_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);render(build(out),out)
