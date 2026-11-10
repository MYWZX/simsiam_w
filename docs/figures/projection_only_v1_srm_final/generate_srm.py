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
 d.shape(G,x,y,2.0,1.1,'vision','feat',name='Predictor_body_'+suffix,width=1.2)
 d.txt(G,x+.08,y+.08,1.84,.36,'Predictor',20,True,name='Predictor_caption_'+suffix)
 math(G,x+.08,y+.48,1.84,.48,'h_{'+suffix+'}',29,'Predictor_symbol_'+suffix)


def sg(sh,x,y,arg):
 d.shape(sh,x,y,2.0,1.1,'white','train',name='Stop_gradient_body_'+arg,width=1.2)
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
 p=Presentation();p.slide_width=Inches(16);p.slide_height=Inches(9.35);s=p.slides.add_slide(p.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white')
 G=d.group(s.shapes,'SRM_Publication_Figure').shapes
 d.txt(G,.3,.16,15.4,.57,'SimSiam Relation Module (SRM)',30,True,name='Figure_title')
 d.txt(G,.4,.80,15.2,.36,'Symmetric stop-gradient matching of projected 256-D features',21,color='muted',name='Figure_scope')
 d.shape(G,1.55,1.30,12.9,4.86,'F7FAFC','border',name='SRM_boundary',width=1.1)
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
  d.shape(G,6.65,y,2.70,1.10,'white','feat',name=name,width=1.3)
  math(G,6.72,y+.12,2.56,.84,formula,25,name+'_math')
 d.edge(G,[(10.90,2.25),(9.35,2.25)],name='Detached_v_to_D')
 d.edge(G,[(5.10,5.20),(6.65,5.20)],name='Detached_u_to_D')
 d.shape(G,7.30,3.325,1.40,.80,'white','feat',name='Symmetric_average',width=1.3)
 math(G,7.30,3.325,1.40,.80,'½Σ',29,'Average_symbol')
 d.edge(G,[(8.0,2.80),(8.0,3.325)],name='Upper_term_to_average')
 d.edge(G,[(8.0,4.65),(8.0,4.125)],name='Lower_term_to_average')
 d.edge(G,[(8.70,3.725),(10.98,3.725)],'supervision',name='Relation_loss_output')
 math(G,11.05,3.34,1.80,.78,'L_{SRM}',29,'Relation_loss_symbol')
 d.txt(G,10.97,4.06,2.03,.29,'relation loss',17,color='muted',name='Relation_loss_caption')
 # One native, editable equation and a compact call map: ready for a paper panel.
 math(G,.50,6.30,15.0,.56,'L_{SRM}=½[D(p_{u},sg(v))+D(p_{v},sg(u))]',25,'Full_SRM_equation')
 # A four-column, understated call strip rather than a second flow diagram.
 T=d.group(s.shapes,'Four_SRM_Call_Signatures').shapes
 d.shape(T,.40,7.10,15.20,1.34,'white','border',name='Call_map_boundary',width=.9)
 calls=[('1 · I–I','(v_{1},v_{2})','h_{I} / h_{I}'),('2 · I–Prompt','(a^{c},t_{P})','h_{I} / h_{T}'),('3 · I–Answer','(a^{c},t_{A})','h_{I} / h_{T}'),('4 · Prompt–Answer','(t_{P},t_{A})','h_{T} / h_{T}')]
 for i,(title,pair,preds) in enumerate(calls):
  x=.40+3.80*i
  if i:d.edge(T,[(x,7.10),(x,8.44)],name='Call_column_'+str(i),arrow=False)
  d.txt(T,x+.08,7.20,3.64,.32,title,20,True,name='Call_title_'+str(i))
  math(T,x+.08,7.59,3.64,.35,'(u,v)='+pair,21,'Call_features_'+str(i))
  math(T,x+.08,8.02,3.64,.31,'(h_{m},h_{n})=('+preds.replace(' / ', ',')+')',18,'Call_predictors_'+str(i))
 d.txt(G,.45,8.65,15.1,.30,'sg: stop-gradient     ℓ₂: L2 normalization     D: negative cosine     Predictor: 256 → 128 → 256',18,color='muted',name='Legend')
 d.txt(G,.45,9.02,15.1,.25,'Predictors are shared by modality across the four calls. PI/PT remain in the shared representation stem.',17,color='muted',name='Sharing_and_module_boundary')
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
 report=dict(pixels=[pix.width,pix.height],canvas_inches=[16,9.35],editable_objects=counts,pdf_raster_images=len(page.get_images()),out_of_page_text=0,fonts=sorted(set(f[3] for f in page.get_fonts())),source='Native editable PPTX -> LibreOffice PDF -> PNG',sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ppt,out/'SRM_publication.pdf',out/'SRM_publication.png']})
 (out/'render_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);render(build(out),out)
