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

