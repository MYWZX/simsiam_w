from pathlib import Path
import subprocess,shutil,json
import fitz
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN
from types import SimpleNamespace
import drawing_primitives as d
import generate_srm as sr
H=Path(__file__).resolve().parent
P=Presentation();P.slide_width=Inches(20);P.slide_height=Inches(10)
s=P.slides.add_slide(P.slide_layouts[6]);G=d.group(s.shapes,'Image_to_Prompt_direction').shapes

def txt(x,y,w,h,t,size=19,bold=False,c='ink'):
 return d.txt(G,x,y,w,h,t,size,bold,c,PP_ALIGN.CENTER,t[:60])
def box(x,y,w,h,t,sub,fill):
 g=d.group(G,t).shapes;d.shape(g,x,y,w,h,fill,'border',name=t+'_body')
 d.txt(g,x+.04,y+.15,w-.08,.36,t,18,True,'ink',name=t+'_title')
 d.txt(g,x+.04,y+.65,w-.08,h-.68,sub,14,False,'muted',name=t+'_detail')
def edge(pts,name,c='line',dash=False):
 o=d.edge(G,pts,'flow',name)
 for seg in o.shapes:
  if hasattr(seg, 'line'):
   seg.line.color.rgb=d.rgb(c)
   if dash:
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    seg.line.dash_style=MSO_LINE_DASH_STYLE.DASH
 return o

txt(.4,.20,19.2,.6,'SRM: Image predicts Prompt',31,True)
txt(.5,.96,19,.4,'One direction of the image–prompt pair · projection_only_v1 baseline',20,c='muted')
d.shape(G,.35,1.65,15.65,2.45,'EAF7F5','border',name='Prediction_path_panel')
d.shape(G,.35,5.18,15.65,2.45,'F3EFF8','border',name='Target_path_panel')
txt(.6,1.79,14.7,.32,'PREDICTION PATH  ·  parameters receive gradients from this term',18,True,'398D9B')
# Each box includes its actual normalization position.
image=[(.60,1.7,'Clean image','224 × 224 × 3','vision'),(2.65,2.30,'CLIP image encoder','768-D → L2 norm','vision'),(5.30,2.15,'Global projection','Linear 768 → 256\ng: 256-D','proj'),(7.80,1.65,'PI','256 → 512 → 512 → 256\nL2 norm','proj'),(9.80,1.65,'AG','Residual adapter\nL2 norm → u','amod'),(11.80,3.85,'Image Predictor hI','256 → 128 → 256\nBN + ReLU; output L2 norm','proj')]
for x,w,t,sub,c in image:box(x,2.35,w,1.43,t,sub,c)
for a,b in zip(image,image[1:]):edge([(a[0]+a[1],3.06),(b[0],3.06)],a[2]+'_to_'+b[2])
txt(12.0,3.83,3.45,.24,'prediction pu: 256-D',14,c='muted')
# Text target uses the same learned context encoder as Answer, but is detached here.
txt(.6,5.32,14.7,.32,'TARGET PATH  ·  no gradients from this term after stop-gradient',18,True,'8965AA')
text=[(.60,1.7,'Prompt','Text string','text'),(2.65,2.30,'CLIP text encoder','[SOT, C1…C5, words, EOT]\nfrozen weights; 768-D + L2','text'),(5.30,1.45,'T','Linear\n768 → 256','proj'),(7.10,1.65,'PT','256 → 512 → 512 → 256\nL2 norm','proj'),(9.10,1.45,'AT','Residual\nadapter','amod'),(10.90,2.1,'WT','Linear 256 → 256\nL2 norm → v','proj'),(13.40,2.25,'sg(v)','Stop-gradient\ntarget: 256-D','white')]
for x,w,t,sub,c in text:box(x,5.87,w,1.43,t,sub,c)
for a,b in zip(text,text[1:]):edge([(a[0]+a[1],6.58),(b[0],6.58)],a[2]+'_to_'+b[2])
# Loss receives both feature vectors; it is not a score head.
box(16.65,3.64,2.95,1.62,'Negative cosine','D = −cos(pu, sg(v))\nscalar training loss','qmod')
edge([(15.65,3.06),(17.1,3.06),(17.1,3.64)],'Prediction_to_loss')
edge([(15.65,6.58),(18.1,6.58),(18.1,5.26)],'Detached_target_to_loss')
# Explicit reverse gradient along a separate lane.
edge([(16.65,4.45),(3.80,4.45),(3.80,3.78)],'Backprop_to_image_path','C56661',True)
txt(4.15,4.54,11.70,.30,'BACKPROP: update hI, AG, PI, global projection and image encoder',16,True,'C56661')
txt(.65,7.92,18.7,.48,'sg blocks this loss from updating C1…C5, T, PT, AT and WT on the target path.',20,True)
txt(.65,8.58,18.7,.57,'Those text parameters can still learn from the reverse direction (Prompt predicts Image) and other losses.\nCLIP text weights stay frozen in every direction; five context vectors are shared by Prompt and Answer.',18,c='muted')
txt(.65,9.48,18.7,.28,'Solid arrows: forward flow    Red dashed arrow: gradient direction    L2 norm: vector normalization    No local P0, QA or final scoring head in this path.',14,c='muted')
s.notes_slide.notes_text_frame.text='Source: models/two_stage_models.py FrozenCLIPTextEncoder.forward, _project_align_image, _align_from_global_and_patch, _project_text, negative_cosine. Baseline trainable visual encoder; frozen-image ablations retain the same forward structure but do not update CLIP image weights. Per-pair directional term shown; symmetric pair loss additionally averages reverse term and applies sample alignment weights.'
sr.attach(SimpleNamespace(slides=[s]));P.save(H/'SRM_Image_Predicts_Prompt.pptx')
subprocess.run([shutil.which('soffice'),'--headless','--convert-to','pdf','--outdir',str(H),str(H/'SRM_Image_Predicts_Prompt.pptx')],check=True,timeout=120)
doc=fitz.open(H/'SRM_Image_Predicts_Prompt.pdf');p=doc[0]
for width,suffix in [(6000,'.png'),(1800,'_preview.png')]:p.get_pixmap(matrix=fitz.Matrix(width/p.rect.width,width/p.rect.width),alpha=False).save(H/('SRM_Image_Predicts_Prompt'+suffix))
assert not p.get_images() and '\ufffd' not in p.get_text()
(H/'render_audit.json').write_text(json.dumps({'pixels':[6000,3000],'pdf_raster_images':len(p.get_images()),'ppt_is_native_editable':True},indent=2))
