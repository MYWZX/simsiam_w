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
 d.shape(G,x,y,2.0,1.1,'FFE0C9','E8BA94',name='Predictor_body_'+suffix,width=1.2)
 d.txt(G,x+.08,y+.08,1.84,.36,'Predictor',20,True,name='Predictor_caption_'+suffix)
 math(G,x+.08,y+.48,1.84,.48,'h_{'+suffix+'}',29,'Predictor_symbol_'+suffix)


def sg(sh,x,y,arg):
 d.shape(sh,x,y,2.0,1.1,'F3F5F8','A9B5C4',name='Stop_gradient_body_'+arg,width=1.2)
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
 p=Presentation();p.slide_width=Inches(16);p.slide_height=Inches(9.4)
 s=p.slides.add_slide(p.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white')
 G=d.group(s.shapes,'SRM_Clear_Overview').shapes
 d.txt(G,.5,.18,15,.58,'SRM | Symmetric feature matching',30,True,align=PP_ALIGN.LEFT,name='Title')
 d.txt(G,.52,.87,15,.36,'Predict the other feature → swap roles → average the two losses.',21,color='muted',align=PP_ALIGN.LEFT,name='Subtitle')
 for idx,(top,source,target,pred) in enumerate([(1.55,'u','v','m'),(3.85,'v','u','n')]):
  R=d.group(G,'Direction_'+str(idx+1)).shapes
  d.shape(R,.50,top,11.2,2.07,'F8FAFC','border',name='Direction_panel_'+str(idx))
  d.txt(R,.72,top+.10,4.4,.35,str(idx+1)+'  Predict '+target+' from '+source,22,True,align=PP_ALIGN.LEFT,name='Direction_title_'+str(idx))
  y=top+1.35
  # The target is shown directly above the match operation, with a separate stopped path.
  d.shape(R,5.60,top+.20,.62,.55,'E5EFF8','528CB3',MSO_SHAPE.OVAL,'Target_feature_'+target)
  math(R,5.60,top+.20,.62,.55,target,26,'Target_symbol_'+target)
  d.edge(R,[(6.22,top+.475),(7.03,top+.475)],name='Target_to_sg_'+target)
  d.shape(R,7.03,top+.16,1.97,.63,'F3F5F8','A9B5C4',name='SG_'+target)
  d.txt(R,7.06,top+.19,1.91,.26,'Stop gradient',16,True,color='687386',name='SG_caption_'+target)
  math(R,7.06,top+.46,1.91,.27,'sg('+target+')',18,'SG_formula_'+target)
  d.edge(R,[(9.00,top+.475),(10.20,top+.475),(10.20,top+.99)],name='Target_to_match_'+target)
  d.shape(R,.92,y-.33,.66,.66,'E5EFF8','528CB3',MSO_SHAPE.OVAL,'Source_feature_'+source)
  math(R,.92,y-.33,.66,.66,source,27,'Source_symbol_'+source)
  d.edge(R,[(1.58,y),(2.12,y)],name='Source_to_predictor_'+source)
  d.shape(R,2.12,y-.40,2.32,.80,'FFE0C9','E8BA94',name='Predictor_'+pred)
  d.txt(R,2.16,y-.32,2.24,.31,'Predictor',21,True,name='Predictor_caption_'+pred)
  math(R,2.16,y+.02,2.24,.29,'h_{'+pred+'}',21,'Predictor_math_'+pred)
  d.edge(R,[(4.44,y),(5.12,y)],name='Predictor_to_L2_'+source)
  d.shape(R,5.12,y-.27,.60,.54,'FFFFFF','border',name='L2_'+source)
  d.txt(R,5.12,y-.27,.60,.54,'L2',17,True,name='L2_caption_'+source)
  d.edge(R,[(5.72,y),(9.30,y)],name='Prediction_to_match_'+source)
  math(R,6.35,y-.50,1.40,.37,'p_{'+source+'}',26,'Prediction_'+source)
  d.txt(R,7.38,y-.49,1.71,.32,'prediction',16,color='muted',name='Prediction_caption_'+source)
  d.shape(R,9.30,y-.36,1.80,.72,'EDE9F7','8B79B1',name='Match_'+source)
  d.txt(R,9.34,y-.30,1.72,.31,'Match',21,True,color='69548E',name='Match_caption_'+source)
  d.txt(R,9.34,y+.04,1.72,.25,'− cosine',15,color='69548E',name='Match_distance_'+source)
  # No shared wires cross between the two directions.
  d.edge(G,[(11.10,y),(12.10,y),(12.10,3.90),(12.65,3.90)],name='Distance_to_average_'+source)
 d.shape(G,12.65,3.37,2.82,1.06,'EDE9F7','8B79B1',name='Average')
 d.txt(G,12.72,3.47,2.68,.34,'3  Average',23,True,color='69548E',name='Average_title')
 d.txt(G,12.72,3.91,2.68,.28,'both directions',17,color='69548E',name='Average_caption')
 d.edge(G,[(14.06,4.43),(14.06,4.97)],name='Average_to_loss')
 math(G,12.7,5.04,2.73,.50,'L_{SRM}',30,'Final_loss')
 d.txt(G,12.65,5.64,2.82,.30,'relation loss',18,True,color='muted',name='Loss_caption')
 d.txt(G,.66,6.13,11.0,.35,'Same inputs, opposite roles. Only the prediction path receives gradients.',17,color='muted',align=PP_ALIGN.LEFT,name='Core_explanation')
 d.txt(G,.52,6.85,15,.33,'USE THE SAME MODULE FOR FOUR RELATIONS',19,True,align=PP_ALIGN.LEFT,name='Call_section_title')
 calls=[('Image ↔ Image','(v_{1},v_{2})','h_{I},h_{I}'),('Image ↔ Prompt','(a^{c},t_{P})','h_{I},h_{T}'),('Image ↔ Answer','(a^{c},t_{A})','h_{I},h_{T}'),('Prompt ↔ Answer','(t_{P},t_{A})','h_{T},h_{T}')]
 for i,(title,pair,preds) in enumerate(calls):
  x=.50+3.82*i;C=d.group(G,'Call_'+str(i+1)).shapes
  d.shape(C,x,7.36,3.58,1.13,'FFFFFF','border',name='Card_'+str(i))
  d.shape(C,x,7.36,3.58,.38,'E5EFF8',None,name='Card_header_'+str(i))
  d.txt(C,x+.05,7.39,3.48,.29,title,18,True,name='Call_title_'+str(i))
  d.txt(C,x+.05,7.80,1.85,.22,'Input pair',13,color='muted',name='Input_caption_'+str(i))
  d.txt(C,x+1.94,7.80,1.59,.22,'Predictors',13,color='muted',name='Predictor_caption_'+str(i))
  math(C,x+.05,8.07,1.85,.30,pair,22,'Input_pair_'+str(i))
  math(C,x+1.94,8.07,1.59,.30,'('+preds+')',18,'Predictor_pair_'+str(i))
 d.txt(G,.52,8.68,15,.27,'Only two predictor networks: one for images and one for text, reused across all four relations.',16,color='muted',name='Predictor_sharing')
 d.txt(G,.52,9.06,15,.25,'Current mode: positive matching only · 256-D inputs · PI/PT are outside SRM · Targets are detached, not permanently frozen',15,color='muted',name='Legend')
 attach(p)
 s.notes_slide.notes_text_frame.text='Projection_only_v1. All four calls reuse exactly two predictor networks by modality. Inputs and weights follow the source-verified previous SRM figure. Each predictor is Linear256->128, BN, ReLU, Linear128->256 then L2 normalization. D is negative cosine. Direction1 D(norm(h_m(u)),sg(v)); direction2 D(norm(h_n(v)),sg(u)); final average 0.5*(D1+D2). Shared input symbols are repeated for readability, not different feature computations. Per-sample weights and batch reduction outside SRM: I-I 1, I-Prompt human a, I-Answer CLIP-derived c, Prompt-Answer a*c. External lambda coefficients not depicted. Extra local/word supervision is separate. No negative push, cross-attention or fusion z. PI/PT outside module. v1/v2=norm PI(g1/g2); a^c=norm AG(norm PI(gclean)); tP=norm WT(AT(norm PT(T(Etext(P))))); tA=norm PT(T(Etext(A))).'
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
 report=dict(pixels=[pix.width,pix.height],canvas_inches=[16,9.4],editable_objects=counts,pdf_raster_images=len(page.get_images()),out_of_page_text=0,fonts=sorted(set(f[3] for f in page.get_fonts())),source='Native editable PPTX -> LibreOffice PDF -> PNG',sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ppt,out/'SRM_publication.pdf',out/'SRM_publication.png']})
 (out/'render_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);render(build(out),out)
