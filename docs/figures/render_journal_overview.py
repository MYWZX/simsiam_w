"""Deterministic vector diagram, explicit coordinates and connections; no model edits."""
from pathlib import Path
from html import escape
import cairosvg
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1900" height="1280" viewBox="0 0 1900 1280">', '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#40546b"/></marker></defs>', '<rect width="1900" height="1280" fill="white"/>']
def text(x,y,lines,size=18,color='#162a43',anchor='middle',bold=False):
 for i,line in enumerate(lines.split('\n')):
  s.append(f'<text x="{x}" y="{y+i*(size+6)}" text-anchor="{anchor}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(line)}</text>')
def box(x,y,w,h,label,fill='#edf3fb',stroke='#567394',size=18):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
 n=len(label.split('\n'));text(x+w/2,y+h/2-(n-1)*(size+6)/2+size*.35,label,size)
def edge(points,label='',lx=0,ly=0,dash=False):
 pts=' '.join(f'{x},{y}' for x,y in points)
 s.append(f'<polyline points="{pts}" fill="none" stroke="#40546b" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
 if label:text(lx,ly,label,15)
def panel(x,y,w,h,title,color):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="white" stroke="{color}" stroke-width="2"/>')
 text(x+20,y+31,title,22,color,'start',True)
teal='#e6f5f0';amber='#fff3df';purple='#f1ebfa'
text(950,36,'Quality–Alignment Decoupled Learning for AIGC Image Assessment',29,bold=True)
panel(20,60,1860,380,'a  Quality-aware multimodal pretraining','#345879')
box(45,123,170,110,'SafeAug 1\nSafeAug 2\nClean view')
box(250,113,280,140,'Shared visual backbone\nCLIP ViT-L/14 + ResNet-50\nSemantic–Structure Fusion')
edge([(215,178),(250,178)])
box(575,113,330,140,'Shared quality projector P_Q\nQuality predictor D_Q\nI–I symmetric SimSiam',teal)
edge([(530,143),(575,143)],'z₁, z₂',550,128)
box(955,113,340,140,'Clean quality embedding q_c\nComponent anchors + ordinal\nFive-level quality text CE',teal)
edge([(530,228),(550,228),(550,275),(735,275),(735,253)],'z_c → same P_Q',675,298)
edge([(905,183),(955,183)],'q_c',930,168)
box(1350,113,495,140,'Separate Clean scoring paths\nQuality MOS + Artifact / Aesthetics\nGlobal / local Alignment supervision',amber)
edge([(530,228),(550,228),(550,320),(1595,320),(1595,253)],'Clean backbone features; not P_Q output',1500,345)
box(45,335,485,75,'Prompt / Answer / region phrases\nFrozen CLIP text + learnable contexts',purple)
box(575,335,720,75,'Independent P_I / D_I and P_T / D_T; relation-specific adapters\nImage–Prompt · Image–Answer · Prompt–Answer · region–word',purple, size=17)
edge([(530,372),(575,372)])
edge([(390,253),(390,285),(560,285),(560,350),(575,350)])
text(1580,381,'P_Q → Quality Adapter initialization',18,'#7548a3',bold=True)
text(1580,408,'Semantic projectors / predictors discarded',16,'#7548a3')
panel(20,460,1860,730,'b  Downstream assessment · Large CLIP · ResNet layer3 · two outputs','#345879')
# Shared feature extraction
box(45,545,150,70,'Image\n224 × 224')
box(225,525,250,80,'CLIP visual ViT-L/14\nglobal feature: 768-D')
box(225,640,250,100,'ResNet-50 layer3\n1024 × 14 × 14\n1×1 proj → 256 × 14 × 14')
edge([(195,580),(210,580),(210,565),(225,565)])
edge([(195,580),(210,580),(210,690),(225,690)])
box(225,790,250,95,'Semantic–Structure Fusion\nglobal query / local K,V\nz: 256-D')
edge([(350,605),(350,620),(500,620),(500,810),(475,810)])
edge([(350,740),(350,790)])
box(225,945,250,80,'Shared local projection L\nP₀: 196 × 256')
edge([(475,690),(515,690),(515,985),(475,985)])
box(45,1060,430,90,'Prompt + shared learnable contexts\nFrozen CLIP text encoder → U (768→256)',purple)
text(260,1170,'Visual / Fusion / local L shared by Q and A',15)
# Quality paths, labelled ports avoid crossing with alignment
panel(550,510,1305,330,'Perceptual quality','#248b7b')
box(575,575,130,65,'z',teal)
box(740,555,230,80,'Quality Adapter\nfrom pretrained P_Q',teal)
edge([(705,595),(720,595),(720,595),(740,595)])
box(1010,555,425,80,'Residual feature → global quality head\nr = z + α · sg(‖z‖) · normalize(a)',teal,size=17)
edge([(970,595),(1010,595)],'a',990,580)
edge([(640,640),(640,655),(1220,655),(1220,635)],'z bypass · α learned, init 0.1',1030,677)
box(575,715,395,85,'P₀ + projected NSS(image)\nPatch scores + NSS weighting\nWeighted aggregation → Q_local',teal,size=17)
box(1480,600,345,145,'Scalar quality fusion\nQ = (1−β) Q_global + β Q_local\nβ: learned scalar\nOUTPUT: PERCEPTUAL QUALITY',teal,size=17)
edge([(1435,595),(1460,595),(1460,625),(1480,625)],'Q_global',1490,575)
edge([(970,758),(1450,758),(1450,710),(1480,710)],'Q_local',1200,746)
# Alignment paths
panel(550,865,1305,300,'Text–image alignment','#b77923')
box(575,920,235,80,'z → image adapter A_G\nv: 256-D',amber,size=17)
box(575,1030,235,80,'U(text) → text adapter A_T\nu: 256-D',amber,size=17)
box(860,920,330,80,'Global pair head\n[v, u, |v−u|] → A_global',amber,size=17)
edge([(810,960),(860,960)])
edge([(810,1070),(835,1070),(835,980),(860,980)])
box(860,1030,450,95,'P₀ → independent patch adapter A_P\nX_I / X_T matching(u); attention pooling → e\nLocal pair head [e, u, |e−u|] → A_local',amber,size=17)
edge([(810,1085),(860,1085)],'u',833,1126)
box(1480,940,345,165,'Sample-wise Alignment gate γ\nInputs: v, u, A_global, A_local\nA = γ A_global + (1−γ) A_local\nOUTPUT: TEXT–IMAGE ALIGNMENT',amber,size=17)
edge([(1190,960),(1480,960)],'A_global, v, u',1330,944)
edge([(1310,1080),(1480,1080)],'A_local',1388,1064)
# Reused feature symbols are named ports; make the fused feature bus explicit.
edge([(475,850),(530,850),(530,608),(575,608)], 'z',545,590)
edge([(530,850),(530,945),(575,945)])
edge([(475,985),(540,985),(540,780),(575,780)], 'P₀',550,765)
edge([(475,1105),(530,1105),(530,1090),(575,1090)], 'U(text)',522,1138)
# Auxiliary bank and transfer caption
text(950,1220,'TRAINING ONLY  a → image-to-text map ↔ five text references → CE (λ = 0.005); score blend = 0',17,'#7548a3')
text(950,1250,'Transfer: compatible backbone, contexts, alignment adapters, quality bank; task heads and layer3 local projection newly initialized.',16)
s.append('</svg>');(P/'journal_framework.svg').write_text('\n'.join(s))
cairosvg.svg2png(url=str(P/'journal_framework.svg'), write_to=str(P/'journal_framework.png'), scale=2)
cairosvg.svg2pdf(url=str(P/'journal_framework.svg'), write_to=str(P/'journal_framework.pdf'))
