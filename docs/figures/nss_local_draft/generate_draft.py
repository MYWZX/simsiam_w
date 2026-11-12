from pathlib import Path
import subprocess,shutil,json
import fitz
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import drawing_primitives as d
import generate_srm as sr
HERE=Path(__file__).resolve().parent

def text(sh,x,y,w,h,t,size=22,bold=False,color='ink',left=False):
 return d.txt(sh,x,y,w,h,t,size,bold,color,PP_ALIGN.LEFT if left else PP_ALIGN.CENTER,t.replace('\n','_')[:75])
def block(sh,x,y,w,h,title,sub='',fill='proj',size=23):
 g=d.group(sh,title).shapes;d.shape(g,x,y,w,h,fill,'border',name=title+'_body')
 text(g,x+.07,y+.08,w-.14,.38,title,size,True)
 if sub:text(g,x+.07,y+.51,w-.14,h-.54,sub,size-4,color='muted')
def edge(sh,pts,name,arrow=True):return d.edge(sh,pts,name=name,arrow=arrow)
def feature(sh,x,y,s,dim=''):
 d.shape(sh,x-.34,y-.28,.68,.56,'vision','feat',name=s+'_feature')
 sr.math(sh,x-.33,y-.27,.66,.54,s,25,s+'_symbol')
 if dim:text(sh,x-.92,y+.40,1.84,.36,dim,17,color='muted')
def panel(sh,y,h,title,fill):
 d.shape(sh,.35,y,19.30,h,fill,'border',name=title+'_region')
 text(sh,.60,y+.12,18.8,.40,title,24,True,left=True)

def build(out):
 p=Presentation();p.slide_width=Inches(20);p.slide_height=Inches(12.7)
 s=p.slides.add_slide(p.slide_layouts[6]);G=d.group(s.shapes,'NSS_local_quality_draft').shapes
 text(G,.45,.25,19.10,.62,'NSS-guided local quality branch | project draft',30,True,left=True)
 text(G,.47,.98,19.06,.36,'Statistics branch + deep-feature branch; current MSCN-like implementation, without GGD fitting',20,color='muted',left=True)
 panel(G,1.62,4.03,'(a) From pixels to 196 local statistical descriptors','F1F6FF')
 feature(G,1.22,3.58,'I','224 × 224 × 3')
 block(G,3.25,2.35,3.12,1.04,'MSCN-like map z','gray → local normalization','vision',22)
 block(G,3.25,4.00,3.12,.94,'Gray gradient','horizontal + vertical differences','vision',22)
 edge(G,[(1.56,3.58),(2.46,3.58)],'Image_input');d.dot(G,2.46,3.58)
 edge(G,[(2.46,3.58),(2.46,2.87),(3.25,2.87)],'Image_to_MSCN')
 edge(G,[(2.46,3.58),(2.46,4.47),(3.25,4.47)],'Image_to_gradient')
 block(G,7.90,2.35,4.31,1.04,'Average pooling to 14 × 14','mean(z), mean(z²), mean(|z|)','qmod',22)
 block(G,7.90,4.00,4.31,.94,'Average pooling to 14 × 14','mean(gradient magnitude)','qmod',22)
 edge(G,[(6.37,2.87),(7.90,2.87)],'MSCN_statistics_pool')
 edge(G,[(6.37,4.47),(7.90,4.47)],'Gradient_statistics_pool')
 block(G,14.02,2.77,3.75,1.77,'Build 6-channel descriptor','m₁, m₂, mabs, mgrad\nmabs / √m₂, m₂','text',21)
 edge(G,[(12.21,2.87),(13.12,2.87),(13.12,3.27),(14.02,3.27)],'MSCN_stats_to_descriptor')
 edge(G,[(12.21,4.47),(13.12,4.47),(13.12,4.05),(14.02,4.05)],'Gradient_stats_to_descriptor')
 feature(G,18.96,3.66,'n','196 × 6');edge(G,[(17.77,3.66),(18.62,3.66)],'NSS_descriptors')
 text(G,14.00,4.84,4.78,.35,'Current code: channel 6 repeats channel 2',15,color='A6654F')
 text(G,3.25,5.13,9.00,.30,'224 → 14: average each 16×16 region (256 values). Flatten the 14×14 grid into 196 rows.',16,color='muted')
 panel(G,6.00,5.37,'(b) Use statistics to enhance and weight local features','EDFAFC')
 feature(G,1.38,7.33,'n','reuse n from (a)')
 block(G,3.20,6.84,2.12,.94,'NSS MLP','6 → 256 → 256','proj',21)
 edge(G,[(1.72,7.33),(3.20,7.33)],'NSS_to_embedding');d.dot(G,2.67,7.33)
 feature(G,1.38,8.63,'P_{0}','196 × 256')
 d.op(G,4.20,8.63,'+',.48,'white')
 edge(G,[(1.72,8.63),(3.96,8.63)],'Local_tokens_to_add')
 edge(G,[(4.20,7.78),(4.20,8.39)],'NSS_embedding_to_add')
 block(G,5.66,8.16,2.10,.94,'QL + L2','residual adapter','amod',22)
 edge(G,[(4.44,8.63),(5.66,8.63)],'Augmented_tokens_to_QL')
 feature(G,8.74,8.63,'r','196 × 256');edge(G,[(7.76,8.63),(8.40,8.63)],'Quality_local_tokens')
 block(G,13.35,8.16,2.03,.94,'Weighted pool','Σ wᵢrᵢ + L2','qmod',20)
 edge(G,[(9.08,8.63),(13.35,8.63)],'Token_values_to_pool');d.dot(G,10.52,8.63)
 feature(G,16.28,8.63,'q_{L}','256-D');edge(G,[(15.38,8.63),(15.94,8.63)],'Pooled_local_feature')
 block(G,17.31,8.16,1.22,.94,'Head','sigmoid','vision',20)
 edge(G,[(16.62,8.63),(17.31,8.63)],'qL_to_score_head')
 edge(G,[(18.53,8.63),(18.88,8.63)],'Local_quality_output');feature(G,19.22,8.63,'s_{L}','local score')
 # Raw n enters the weighting network separately from the embedded addition.
 edge(G,[(2.67,7.33),(2.67,8.53)],'Raw_NSS_lower1',False)
 edge(G,[(2.67,8.73),(2.67,10.99),(11.49,10.99),(11.49,10.65)],'Raw_NSS_to_gate')
 block(G,9.99,9.65,3.00,1.00,'NSS gate + Softmax','[r, n]: 262 → 256 → 1','qmod',21)
 edge(G,[(10.52,8.63),(10.52,9.30),(11.49,9.30),(11.49,9.65)],'r_to_gate')
 edge(G,[(12.99,10.15),(14.37,10.15),(14.37,9.10)],'Weights_to_pool')
 text(G,14.51,9.85,4.06,.50,'196 weights w; sum(w) = 1',18,color='muted')
 text(G,5.01,10.51,4.04,.32,'raw 6-D statistics reused',16,color='muted')
 text(G,.50,11.65,19.0,.34,'Pooling weights act on FEATURES r before the score head. They do not multiply the final local score.',20,True,color='muted')
 text(G,.50,12.12,19.0,.28,'Layout inspired by Fig. 2 of Xia et al., TMM 2025; computations follow our code. Small ε terms omitted. sL later joins the global quality score.',16,color='muted')
 sr.attach(p);s.notes_slide.notes_text_frame.text='Project-specific NSS draft. Does not reproduce paper GGD. Current pooled_var is mean(z²), not centered variance; channel6 equals channel2. Grad from original grayscale image. 224224 stats map adaptive avg pool14x14 means 16x16values/cell; each pool point has256values. Ratio computed after pooling, not pooled ratio. n6D pertoken -> nss_proj Linear6->256 GELU Linear256->256, add to P0; QL residual adapter+L2; gate on concat r,n262D ->256->1 then softmax over196; pool r then normalize and score. n repeated with reuse label is same tensor. No cross-attention, no text gate in this Quality branch. Layout only inspired by user-provided paper; no copyrighted figure copied.'
 p.save(out/'NSS_Local_Draft.pptx')
 return p

def main():
 out=HERE;p=build(out)
 subprocess.run([shutil.which('soffice'),'--headless','--convert-to','pdf','--outdir',str(out),str(out/'NSS_Local_Draft.pptx')],check=True,timeout=120)
 pdf=fitz.open(out/'NSS_Local_Draft.pdf');page=pdf[0]
 for width,name in [(6000,'NSS_Local_Draft.png'),(1800,'preview.png')]:page.get_pixmap(matrix=fitz.Matrix(width/page.rect.width,width/page.rect.width),alpha=False).save(out/name)
 assert not page.get_images() and '\ufffd' not in page.get_text()
 spans=[z for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for z in l['spans']]
 assert all(page.rect.contains(fitz.Rect(z['bbox'])) for z in spans)
 (out/'render_audit.json').write_text(json.dumps({'pixels':[6000,3810],'pdf_raster_images':0,'out_of_page_text':0,'source':'Editable PPTX -> PDF -> PNG'},indent=2))

if __name__=='__main__':main()
