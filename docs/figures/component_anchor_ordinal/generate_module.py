"""Native editable compact module + source-faithful worked example."""
from pathlib import Path
import math,json,subprocess,shutil
import fitz
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.enum.dml import MSO_LINE_DASH_STYLE
import drawing_primitives as d
import generate_srm as sr
HERE=Path(__file__).resolve().parent
d.C.update(ink='243447',muted='69788B',border='D3DEE8',line='526174',vision='E0EDFA',proj='ECE5F8',qmod='DAF1F1',amod='FCE8DA',text='F1E8FA',transfer='9673BA')
P=Presentation();P.slide_width=Inches(20);P.slide_height=Inches(10)

def text(sh,x,y,w,h,t,size=22,bold=False,c='ink',left=False,name=None):
 return d.txt(sh,x,y,w,h,t,size,bold,c,PP_ALIGN.LEFT if left else PP_ALIGN.CENTER,name or t.replace('\n','_')[:65])
def formula(sh,x,y,w,h,t,size=25,name='Formula'):return sr.math(sh,x,y,w,h,t,size,name)
def arrow(sh,pts,name,supervision=False,arrow=True):return d.edge(sh,pts,'supervision' if supervision else 'flow',name,arrow)
def box(sh,x,y,w,h,title,sub='',fill='proj',size=22,name=None):
 G=d.group(sh,name or title).shapes;d.shape(G,x,y,w,h,fill,'border',name=(name or title)+'_body',width=1.1)
 if sub:
  text(G,x+.07,y+.09,w-.14,.35,title,size,True)
  text(G,x+.07,y+.52,w-.14,h-.55,sub,size-5,c='muted')
 else:text(G,x+.07,y+.06,w-.14,h-.12,title,size,True)
 return G

def slide(title,sub):
 s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=d.rgb('white');G=d.group(s.shapes,title).shapes
 text(G,.45,.23,19.1,.60,title,30,True,left=True)
 text(G,.47,.95,19.06,.37,sub,20,c='muted',left=True)
 return s,G

def feature(sh,x,y,s):
 d.shape(sh,x-.34,y-.29,.68,.58,'vision','6695BD',MSO_SHAPE.OVAL,'Feature_'+s)
 formula(sh,x-.33,y-.28,.66,.56,s,26,'Feature_label_'+s)

def bars(sh,x,y,w,h,values,color='548EBC',targets=False,nums=False,name='Probabilities'):
 G=d.group(sh,name).shapes;n=len(values);gap=.11;bw=(w-gap*(n-1))/n
 # Separate bars show an actual distribution, never ten-class concatenation.
 for i,v in enumerate(values):
  xx=x+i*(bw+gap)
  d.shape(G,xx,y,bw,h,'F3F6FA','DCE4EB',MSO_SHAPE.RECTANGLE,name+'_background_'+str(i),.6)
  if v>0:d.shape(G,xx,y+h*(1-v),bw,max(.008,h*v),color,None,MSO_SHAPE.RECTANGLE,name+'_bar_'+str(i))
  text(G,xx-.05,y+h+.06,bw+.1,.25,str(i+1),14,c='muted')
  if nums:text(G,xx-.08,y-.30,bw+.16,.25,f'{v:.2f}',14,c='muted')
 return G

def anchors(sh,x,y,w,name,color):
 G=d.group(sh,name+'_fixed_bank').shapes
 text(G,x,y,w,.29,name,18,True,c=color,left=True)
 # Equal angular spacing on an ordered 90 degree arc (2-D schematic of 256-D vectors).
 cx=x+w/2;cy=y+1.00;radius=w*.43
 points=[]
 for a in [-45,-22.5,0,22.5,45]:
  a=math.radians(a);points.append((cx+radius*math.sin(a),cy-radius*.33*math.cos(a)))
 for i,(xx,yy) in enumerate(points):
  if i:arrow(G,[points[i-1],(xx,yy)],name+'_order_'+str(i),arrow=False)
  d.shape(G,xx-.07,yy-.07,.14,.14,color,None,MSO_SHAPE.OVAL,name+'_anchor_'+str(i+1))
  text(G,xx-.18,yy+.12,.36,.23,str(i+1),13,c='muted')
 text(G,x,y+1.05,w,.25,'low label value → high label value',13,c='muted')

def compact():
 s,G=slide('Ordered Component Supervision','Fixed level anchors shape the representation; cumulative probabilities preserve the level order.')
 d.shape(G,.35,1.60,19.30,7.54,'FAFCFE','border',name='Module_boundary')
 # Representation source and separate fixed component banks.
 feature(G,1.18,4.04,'q');text(G,.50,4.50,1.82,.71,'256-D\nfrom qA / qL',17,c='muted')
 d.shape(G,2.20,1.70,3.42,2.22,'FFFFFF','border',name='Two_fixed_anchor_banks')
 anchors(G,2.38,1.80,3.06,'Artifact','398D9B')
 # Second bank is represented by a parallel colored ordered row to avoid a fake shared plane.
 text(G,2.38,3.15,3.06,.25,'Aesthetics · separate bank',15,True,c='BD855E')
 pts=[(2.70+j*.47,3.61+(.05 if j in (0,4) else -.03 if j==2 else 0)) for j in range(5)]
 for j,(xx,yy) in enumerate(pts):
  if j:arrow(G,[pts[j-1],(xx,yy)],'Aesthetic_order_'+str(j),arrow=False)
  d.shape(G,xx-.06,yy-.06,.12,.12,'BD855E',None,MSO_SHAPE.OVAL,'Aesthetic_anchor_'+str(j+1))
 text(G,2.28,4.50,3.26,.42,'5 fixed anchors per component',16,c='muted')
 box(G,6.41,3.51,3.05,1.06,'Cosine + Softmax','compare q with each anchor','vision',22)
 # q routes under the bank; anchors enter from above. No crossed wires.
 arrow(G,[(1.52,4.04),(6.41,4.04)],'q_to_similarities')
 arrow(G,[(5.62,3.02),(7.94,3.02),(7.94,3.51)],'Fixed_banks_to_similarities')
 # Schematic probability bars; computed worked example appears on next page.
 bars(G,10.34,3.39,2.50,.83,[.10,.20,.40,.23,.07],name='Five_level_probability_schematic')
 text(G,10.12,2.83,2.94,.35,'Five-level distribution',18,True)
 formula(G,10.16,4.56,2.86,.42,'p_{c}',24,'Component_probabilities')
 arrow(G,[(9.46,4.04),(10.18,4.04)],'Similarities_to_probabilities')
 # Top branch: categorical placement. Bottom branch: explicit ordinal transform.
 box(G,16.10,2.28,3.00,1.02,'Anchor CE','match the labelled level','qmod',23)
 arrow(G,[(12.99,4.04),(14.34,4.04),(14.34,2.79),(16.10,2.79)],'Probabilities_to_anchor_loss')
 text(G,16.05,1.81,3.10,.31,'1 · Level placement',18,True,c='398D9B')
 d.shape(G,9.99,5.95,3.47,1.50,'proj','border',name='Fixed_ordinal_transform')
 text(G,10.06,6.03,3.33,.31,'Four ordered thresholds',19,True)
 bars(G,10.49,6.49,2.47,.45,[.90,.70,.30,.07],'8B76B6',name='Four_threshold_schematic')
 text(G,10.06,7.21,3.33,.20,'P(level > k)',13,c='muted')
 arrow(G,[(12.99,4.04),(13.65,4.04),(13.65,5.54),(11.73,5.54),(11.73,5.95)],'Probabilities_to_ordinal_transform')
 d.dot(G,13.65,4.04,.035)
 box(G,16.10,6.04,3.00,1.05,'Ordinal BCE','match four ordered thresholds','amod',23)
 arrow(G,[(13.46,6.565),(16.10,6.565)],'Cumulative_to_ordinal_loss')
 text(G,16.05,5.56,3.10,.31,'2 · Level ordering',18,True,c='BD855E')
 # Ground-truth targets are repeated with identical labels, not parameter-sharing edges.
 box(G,16.22,3.84,2.76,.61,'Human label y','',fill='white',size=18,name='Anchor_label_target')
 arrow(G,[(17.60,3.84),(17.60,3.30)],'Anchor_target_supervision',True)
 box(G,16.22,7.64,2.76,.61,'Same label y','',fill='white',size=18,name='Ordinal_label_target')
 arrow(G,[(17.60,7.64),(17.60,7.09)],'Ordinal_target_supervision',True)
 text(G,2.30,6.13,6.70,.96,'One quality feature q; two component labels.\nEach component uses its own fixed anchor bank.',21,c='muted')
 text(G,2.30,7.58,10.20,.49,'The ordinal branch reuses the five probabilities; it is not another learned prediction head.',18,c='muted')
 text(G,.60,8.59,18.80,.30,'c ∈ {artifact, aesthetics} · two losses per component · fixed anchors are constants, not text embeddings',17,c='muted')
 text(G,.48,9.48,19.04,.27,'Pretraining only. Both losses send gradients back to q and its global/local representation paths. Bars are schematic; anchors lie in separate planes.',16,c='muted')
 s.notes_slide.notes_text_frame.text='Compact replacement for QAM Fixed component anchors + ordinal targets. Two independent fixed banks of five256D vectors, distinct orthogonal planes (Fourier1/2), sameq. For eachcomponent p=softmax(q dot A/T), targetlabel interpolated betweenneighbor centers. CE(p,t), ordinalBCE(1-cumsum(p)[:-1],1-cumsum(t)[:-1]). Two branches parallel; ordinal reusesp, no additional network. Point layouts are schematic, not another learned2D projection or clustering. Groundtruthlabels repeated denote same label, no duplicated network. Probability bars illustrative, not model measurements.'


def detail():
 s,G=slide('From one human label to two supervision signals','Worked Artifact example: the same five-level distribution supports both anchor and ordinal losses.')
 # Values are computed from the actual fixed90degree arc and temperature .2.
 angles=[-45,-22.5,0,22.5,45];phi=10
 logits=[math.cos(math.radians(a-phi))/.2 for a in angles];exps=[math.exp(x-max(logits)) for x in logits];p=[x/sum(exps) for x in exps]
 t=[0,0,.6,.4,0];cp=[1-sum(p[:i+1]) for i in range(4)];ct=[1-sum(t[:i+1]) for i in range(4)]
 box(G,.65,1.86,4.27,.85,'Feature q + fixed anchors','cosine similarities / temperature','vision',21)
 arrow(G,[(4.92,2.28),(5.60,2.28)],'Feature_to_five_probabilities')
 bars(G,5.75,2.10,4.40,1.05,p,nums=True,name='Predicted_five_probabilities')
 text(G,5.60,1.56,4.70,.32,'Predicted probabilities p',21,True)
 box(G,.65,5.06,4.27,.85,'Human label: y = 0.70','between center 0.667 and 0.75','white',21)
 arrow(G,[(4.92,5.48),(5.60,5.48)],'Label_to_soft_target',True)
 bars(G,5.75,5.30,4.40,1.05,t,'4A9F9F',nums=True,name='Interpolated_target_distribution')
 text(G,5.60,4.76,4.70,.32,'Soft target t = [0, 0, .6, .4, 0]',20,True,c='398D9B')
 # CE target comparison, using short separated ports.
 box(G,11.27,1.95,3.07,1.02,'Anchor CE','compare p and t','qmod',23)
 arrow(G,[(10.30,2.63),(11.27,2.63)],'Predicted_to_CE')
 arrow(G,[(10.30,5.83),(10.69,5.83),(10.69,2.93),(10.87,2.93)],'Target_to_CE1',True,False)
 arrow(G,[(11.03,2.93),(11.27,2.93)],'Target_to_CE',True)
 d.dot(G,10.95,2.63,.035);d.dot(G,10.58,5.83,.035)
 # The fixed tail-probability conversion is explicit, with true numerical values.
 box(G,11.27,4.02,3.07,.83,'Fixed cumulative sum','P(level > k), k = 1…4','proj',19)
 arrow(G,[(10.30,2.63),(10.95,2.63),(10.95,4.38),(11.27,4.38)],'p_to_cumulative')
 # At crossing with target route use gap, no junction.
 arrow(G,[(10.30,5.83),(10.58,5.83),(10.58,6.73),(12.80,6.73),(12.80,4.85)],'t_to_cumulative',True)
 bars(G,15.11,3.34,3.87,1.03,cp,'7863A8',nums=True,name='Predicted_threshold_probabilities')
 text(G,14.87,2.79,4.33,.32,'Predicted: P(level > k)',19,True,c='7863A8')
 bars(G,15.11,5.97,3.87,1.03,ct,'D89B6D',nums=True,name='Target_threshold_probabilities')
 text(G,14.87,5.40,4.33,.32,'Target: [1, 1, .4, 0]',19,True,c='B57B4D')
 arrow(G,[(14.34,4.29),(14.65,4.29),(14.65,3.88),(15.00,3.88)],'Converted_p_to_thresholds')
 arrow(G,[(14.34,4.62),(14.65,4.62),(14.65,6.49),(15.00,6.49)],'Converted_t_to_thresholds',True)
 box(G,15.70,4.62,2.70,.60,'Ordinal BCE','',fill='amod',size=21)
 arrow(G,[(17.05,4.37),(17.05,4.62)],'Threshold_p_to_BCE')
 arrow(G,[(19.10,6.49),(19.35,6.49),(19.35,4.92),(18.40,4.92)],'Threshold_target_to_BCE',True)
 # Loss summaries avoid another set of long crossing wires.
 text(G,.65,7.48,18.70,.47,'Anchor CE asks “Which level?”     Ordinal BCE asks “Which ordered thresholds should be exceeded?”',23,True)
 text(G,.65,8.24,18.70,.70,'The target is generated from the human score. The predictions come from q.\nBoth losses update the feature extractor; the anchor vectors and cumulative transform remain fixed.',21,c='muted')
 text(G,.48,9.48,19.04,.27,'Example values are rounded, follow the ordered-arc / soft-label formulas, and are not experimental results. ε clipping omitted.',16,c='muted')
 s.notes_slide.notes_text_frame.text=json.dumps(dict(example='Artifact',centers=[.41666667,.58333333,.66666669,.75,.91666669],label=.7,soft_target=t,probabilities=p,ordinal_probabilities=cp,ordinal_target=ct,source='ordered90degree anchors, q projection angle10degrees, temperature.2'),indent=2)
 return p,t,cp,ct


def main():
 compact();values=detail()
 from types import SimpleNamespace
 from pptx.oxml.ns import qn
 for s in P.slides:sr.attach(SimpleNamespace(slides=[s]))
 P.save(HERE/'Component_Anchor_Ordinal.pptx')
 subprocess.run([shutil.which('soffice'),'--headless','--convert-to','pdf','--outdir',str(HERE),str(HERE/'Component_Anchor_Ordinal.pptx')],check=True,timeout=120)
 pdf=fitz.open(HERE/'Component_Anchor_Ordinal.pdf');report=[]
 for i,page in enumerate(pdf):
  name=['01_Compact_Module','02_Worked_Example'][i]
  for width,suffix in [(6000,'.png'),(1800,'_preview.png')]:page.get_pixmap(matrix=fitz.Matrix(width/page.rect.width,width/page.rect.width),alpha=False).save(HERE/(name+suffix))
  assert not page.get_images() and '\ufffd' not in page.get_text()
  spans=[v for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for v in l['spans']]
  assert all(page.rect.contains(fitz.Rect(v['bbox'])) for v in spans)
  ids=[v.get('id') for v in P.slides[i].shapes._spTree.iter(qn('p:cNvPr'))];assert len(ids)==len(set(ids))
  report.append(dict(name=name,pixels=[6000,3000],pdf_raster_images=0,out_of_page_text=0))
 (HERE/'render_audit.json').write_text(json.dumps(report,indent=2))
 (HERE/'example_values.json').write_text(json.dumps(dict(p=values[0],target=values[1],ordinal_p=values[2],ordinal_target=values[3]),indent=2))

if __name__=='__main__':main()
