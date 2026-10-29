"""Deterministic, editable SVG of the requested fused-z I-I model."""
from pathlib import Path
from html import escape

W,H=1760,1390
out=[]
BLUE='#174a9c';PURPLE='#7825c4';TEAL='#147e91';ORANGE='#b95b26';GRAY='#53616d'
def text(x,y,s,size=18,color='#17212b',anchor='start',bold=False):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{"bold" if bold else "normal"}">{escape(s)}</text>')
def box(x,y,w,h,lines,fill='#fff',stroke='#a8b7c6',size=17):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}"/>')
    if isinstance(lines,str):lines=[lines]
    for i,s in enumerate(lines):text(x+w/2,y+h/2-(len(lines)-1)*11+6+i*22,s,size,anchor='middle')
def arrow(x1,y1,x2,y2,color='#25313b',dash=False,role=None):
    attribute=f' id="{role}"' if role else ''
    dash_attribute = 'stroke-dasharray="6 4"' if dash else ''
    out.append(f'<path{attribute} d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="1.6" {dash_attribute} marker-end="url(#{"purple" if color==PURPLE else "black"})"/>')
def route(points,color='#25313b',dash=False,role=None):
    attribute=f' id="{role}"' if role else ''
    dash_attribute = 'stroke-dasharray="6 4"' if dash else ''
    d='M'+' L'.join(f'{x},{y}' for x,y in points)
    out.append(f'<path{attribute} d="{d}" fill="none" stroke="{color}" stroke-width="1.6" {dash_attribute} marker-end="url(#{"purple" if color==PURPLE else "black"})"/>')
def cube(x,y,label,sub='',teal=False):
    col='#24afbf' if teal else '#4b9ee4';light='#a5edf1' if teal else '#a7d4fa'
    out.append(f'<polygon points="{x},{y} {x+9},{y-7} {x+35},{y-7} {x+26},{y}" fill="{light}" stroke="{col}"/>')
    out.append(f'<rect x="{x}" y="{y}" width="26" height="28" fill="{col}" opacity=".85"/>')
    out.append(f'<polygon points="{x+26},{y} {x+35},{y-7} {x+35},{y+21} {x+26},{y+28}" fill="{light}" stroke="{col}"/>')
    if teal:
        for i in range(1,4):
            out.append(f'<path d="M{x+i*6.5},{y} v28 M{x},{y+i*7} h26" stroke="#e4ffff" stroke-width=".6"/>')
    text(x+17,y-15,label,19,anchor='middle',bold=True)
    if sub:text(x+17,y+49,sub,15,GRAY,anchor='middle')
def plus(x,y,symbol="+"):
    out.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#fff" stroke="#25313b" stroke-width="1.6"/>');text(x,y+6,symbol,22,anchor='middle')
def encoder(x,y,w,h,lines,color=BLUE):
    out.append(f'<polygon points="{x},{y} {x+w},{y+15} {x+w},{y+h-15} {x},{y+h}" fill="{color}" fill-opacity=".09" stroke="{color}" stroke-width="1.7"/>')
    if isinstance(lines,str):lines=[lines]
    for i,s in enumerate(lines):text(x+w/2,y+h/2-(len(lines)-1)*12+5+i*24,s,18,color,anchor='middle',bold=True)
def photo(x,y,size=100):
    out.append(f'<g transform="translate({x},{y})"><rect width="{size}" height="{size}" fill="#b8deef" stroke="#aac1d5"/><polygon points="0,55 25,15 45,48 70,8 {size},55" fill="#7289a4"/><polygon points="15,31 25,15 33,32 70,8 83,31 68,24 58,32" fill="#fff"/><rect y="55" width="{size}" height="{size-55}" fill="#78b7c9"/><path d="M25,73 L70,73 L61,84 L33,84 Z" fill="#bc3c29"/><path d="M28,70 L67,70 L70,73 L25,73 Z" fill="#ef7751"/></g>')
def heading(x,y,s,size=24,color='#101c29'):text(x,y,s,size,color,bold=True)

def render():
    out.clear()
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Liberation Serif, Times New Roman, serif">')
    out.append('<defs><marker id="black" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#25313b"/></marker><marker id="purple" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#7825c4"/></marker></defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')
    heading(25,40,'(a) Quality-aware multimodal pretraining — fused-z I–I',30)
    # Shared encoder and fused feature. Named ports avoid false wire joins.
    photo(25,150,95);text(72,133,'Image I',18,anchor='middle',bold=True)
    text(72,266,'224 × 224 × 3',15,anchor='middle')
    arrow(122,198,155,198);box(160,165,105,68,['SafeAug ×2','+ Clean'],size=17)
    arrow(265,198,290,198)
    box(290,75,365,260,[],stroke='#a6bed6')
    heading(305,100,'Shared visual encoders',20)
    encoder(310,120,110,80,['CLIP','ViT-L/14']);arrow(420,160,445,160)
    box(445,141,112,38,'Wg: 768→256',fill='#eef4ff',size=16);arrow(557,160,585,160);cube(587,146,'g','256-D')
    encoder(310,220,110,80,['ResNet-50','layer3'],TEAL);arrow(420,260,445,260)
    box(445,236,112,48,['R: 1024→256','L: 256→256'],fill='#edfbfc',size=15);arrow(557,260,585,260);cube(587,246,'P₀','196 × 256',True)
    text(305,321,'Fl = R(layer3); L and LF are independent',15,GRAY)
    box(300,352,340,65,['Fusion: Q = g; K,V = LF(Fl)','Cross-attention → z (256-D)'],fill='#f5f1fc',stroke='#b8a1da',size=18)
    text(300,440,'Fusion used by I–I and image–text relations.',16,BLUE)
    # Symmetric I-I, distinct projected targets and predictors.
    box(685,75,535,218,[],stroke='#a6bed6');heading(700,103,'Image–Image SimSiam (symmetric)',21)
    for y,i,j in [(143,'1','2'),(223,'2','1')]:
        cube(712,y-4,'z'+i,'fused z');arrow(752,y+10,786,y+10,role='ii-z'+i+'-to-pi')
        encoder(790,y-18,66,55,['PI'],PURPLE);arrow(856,y+10,882,y+10)
        cube(886,y-4,'h'+i);arrow(924,y+10,952,y+10)
        encoder(956,y-18,66,55,['DI'],ORANGE);arrow(1022,y+10,1053,y+10)
        text(1062,y+16,'p'+i+' vs sg(h'+j+')',19)
    text(821,198,'shared PI / DI',15,PURPLE,anchor='middle')
    # Image-text and text paths.
    box(1250,75,480,218,[],stroke='#a6bed6');heading(1266,103,'Image–text relations',21)
    box(1268,122,442,45,'z → PI → AG → DI  ↔  text targets',fill='#faf6ff',size=19)
    text(1268,196,'Prompt: T* → U → PT → AT → DT',18)
    text(1268,229,'Answer: T* → U → PT → DT',18)
    text(1268,264,'T*: frozen text tower + trainable contexts',15,GRAY)
    box(685,320,1045,98,[],stroke='#c7b2e3')
    heading(703,348,'Four SimSiam relations',21,PURPLE)
    text(703,377,'I–I   ·   Image–Prompt   ·   Image–Answer   ·   Prompt–Answer',22)
    text(703,402,'Positive local / phrase supervision retained; negative push disabled.',16,GRAY)
    # Clean quality paths explicitly separate, converge only AFTER encoders.
    box(25,465,1070,125,[],stroke='#b6c9d9');heading(40,491,'Clean quality supervision — two independent paths',20)
    text(42,531,'g',20,BLUE);arrow(65,525,100,525);box(102,506,112,38,'PQ → qg',fill='#f6efff',size=18)
    arrow(214,525,400,525,role='clean-qg-to-feature-mix')
    text(42,572,'P₀ + WNSS(n)',18,TEAL);arrow(185,566,214,566);box(216,548,145,35,'QL + pool → ql',fill='#effbfc',size=17)
    route([(361,566),(390,566),(390,550),(400,550)],role='clean-ql-to-feature-mix')
    box(400,512,350,60,['q = norm[(1−η)qg + ηql]','feature mixture (not g + P₀)'],fill='#fff',size=18)
    arrow(750,542,780,542);box(785,512,285,60,['Component anchors + ordinal','Artifact / Aesthetics'],fill='#faf5ff',size=18)
    box(1120,465,610,125,[],stroke='#e3baa2');heading(1137,492,'Separate Clean MOS supervision',20,ORANGE)
    text(1137,527,'Artifact/Aesthetics heads: [qg, ql, |qg−ql|]',19)
    text(1137,561,'Quality / Align heads supervise Clean score outputs.',18)
    arrow(860,598,860,625,PURPLE,True)
    text(885,616,'Transfer encoders, PQ→QA, PI/PT, adapters and local quality; reset score heads.',17,PURPLE)
    out.append('<path d="M25,637 H1730" stroke="#8795a2" stroke-dasharray="6 5"/>')
    heading(25,676,'(b) Downstream quality and alignment assessment',30)
    # Downstream backbone. No cross-attention scoring branch.
    box(25,702,365,535,[],stroke='#b6c9d9');heading(40,731,'Visual encoders (fine-tuned)',20)
    photo(45,850,85);text(88,952,'Image I',17,anchor='middle')
    route([(130,890),(150,890),(150,810),(175,810)])
    route([(150,890),(150,1070),(175,1070)])
    encoder(175,765,105,90,['CLIP','ViT-L/14']);arrow(280,810,301,810);cube(315,795,'g','768→256')
    encoder(175,1025,105,90,['ResNet-50','layer3'],TEAL);arrow(280,1070,301,1070);cube(315,1055,'P₀','196 × 256',True)
    text(43,1160,'ResNet: 1024→256 → Fl',17,TEAL)
    text(43,1187,'Local L: 256→256 → P₀',17,TEAL)
    text(43,1215,'Repeated ports denote the same tensor.',15,GRAY)
    # Quality scoring: separate lanes, named ports for residual/statistics.
    box(420,702,1310,310,[],fill='#f0fbfc',stroke='#afdce0');heading(438,730,'Quality branch Q',24,TEAL)
    text(442,754,'Global',17,TEAL,bold=True)
    cube(462,790,'g');arrow(501,804,528,804)
    box(530,777,265,56,['QA (from PQ)','256→512→512→256'],fill='#e0f7fa',stroke='#80bfc7',size=18)
    arrow(795,804,820,804);cube(826,790,'qg')
    arrow(867,804,895,804);box(900,781,197,48,'α·sg(||g||)·norm(qg)',size=17)
    arrow(1097,804,1122,804);plus(1138,804)
    box(1110,746,56,29,'g',fill='#edf4ff',size=17);arrow(1138,776,1138,788,role='quality-global-g-residual')
    arrow(1154,804,1200,804);box(1205,781,165,48,'Hg + sigmoid',size=18)
    arrow(1370,804,1400,804);cube(1403,790,'Qg')
    text(442,846,'Local',17,TEAL,bold=True)
    cube(462,884,'P₀','196 × 256',True);arrow(501,898,545,898);plus(560,898)
    box(530,847,60,28,'sᵢ',fill='#edf6ff',size=17);arrow(560,876,560,882,role='quality-local-statistics-add')
    arrow(576,898,613,898);box(615,876,154,44,'QL + norm',size=18);arrow(769,898,800,898);cube(806,884,'oᵢ')
    arrow(844,898,875,898);box(880,868,195,62,['Gate MLP([oᵢ,nᵢ])','softmax → wᵢ'],size=18)
    arrow(1075,898,1102,898);text(1110,866,'inputs: wᵢ, oᵢ',15,TEAL);box(1105,875,218,47,'ql = norm Σᵢ wᵢoᵢ',size=19)
    arrow(1323,898,1345,898);box(1350,877,117,43,'Hl + sigmoid',size=16);arrow(1467,898,1484,898);cube(1489,884,'Ql')
    # Single NSS extractor, no shared wires to either score head.
    box(442,949,980,48,[],stroke='#bbaddb')
    text(452,980,'Image I',17);arrow(515,974,540,974)
    box(544,955,194,35,'NSS-like statistics',fill='#f6f1fc',size=17)
    arrow(738,974,763,974);text(770,981,'nᵢ (6-D)',18)
    arrow(843,974,869,974);box(873,955,224,35,'WNSS MLP: 6→256',fill='#f6f1fc',size=18)
    arrow(1097,974,1124,974);text(1134,981,'sᵢ (256-D)',18)
    text(1237,981,'reuse nᵢ in gate',15,PURPLE)
    text(1510,760,'Q = (1−β)Qg + βQl',19,BLUE)
    route([(1440,804),(1550,804),(1550,843)])
    route([(1525,898),(1550,898),(1550,875)])
    plus(1550,859,"β");arrow(1566,859,1600,859)
    box(1603,834,110,52,['Quality Q'],fill='#d1f2f5',stroke=TEAL,size=21)
    # Alignment paths and scoring heads.
    box(420,1030,1310,288,[],fill='#fff6ed',stroke='#e6c5a8');heading(438,1060,'Alignment branch A',24,ORANGE)
    for y,label,adapter,output in [(1100,'g','AG','v'),(1180,'P₀','AP','pᵢ')]:
        text(445,y+6,label,20,BLUE);arrow(480,y,510,y)
        encoder(514,y-24,60,48,'PI',PURPLE);arrow(574,y,600,y)
        encoder(604,y-24,60,48,adapter,ORANGE);arrow(664,y,691,y)
        cube(695,y-14,output)
    text(545,1145,'shared PI',15,PURPLE,anchor='middle')
    text(442,1242,'Prompt + 4 contexts',16,PURPLE)
    box(440,1254,105,42,['T* frozen'],fill='#f7f0ff',size=17);arrow(545,1275,560,1275)
    box(565,1254,85,42,'U',size=19);arrow(650,1275,665,1275)
    box(670,1254,55,42,'PT',fill='#f7f0ff',size=19);arrow(725,1275,741,1275)
    box(746,1254,55,42,'AT',fill='#fff0e3',size=19);arrow(801,1275,822,1275);text(833,1282,'u',21,PURPLE)
    arrow(734,1100,786,1100)
    arrow(734,1180,786,1180)
    box(790,1079,155,43,'[v,u,|v−u|]',size=19);arrow(945,1100,968,1100)
    box(973,1079,202,43,'Global MLP + sigmoid',size=17);arrow(1175,1100,1200,1100);cube(1207,1086,'Ag')
    box(790,1153,240,61,['pᵢ,u → XI(pᵢ)·XT(u)','softmax → aᵢ'],size=17)
    arrow(1030,1183,1050,1183);box(1055,1153,170,61,['e = Σᵢ aᵢpᵢ','pool original pᵢ'],size=18)
    arrow(1225,1183,1240,1183);box(1245,1153,164,61,['[e,u,|e−u|]','MLP + sigmoid'],size=17)
    arrow(1409,1183,1430,1183);cube(1438,1169,'Al')
    box(1490,1072,222,55,['γ = sigmoid G','([v,u,Ag,Al])'],size=18)
    text(1493,1142,'inputs: Ag, Al, γ',15,ORANGE)
    box(1488,1150,224,76,['A = γAg + (1−γ)Al','Text–image alignment'],fill='#ffe9d6',stroke=ORANGE,size=18)
    arrow(1600,1127,1600,1147)
    text(900,1277,'Named ports v,u,pᵢ,Ag,Al avoid crossed wires; γ is sample-wise.',16,GRAY)
    # Footer explains state succinctly without further protocol clutter.
    text(30,1350,'Solid arrows: forward   ·   Purple dashed: transfer/shared weights   ·   sg: stop-gradient   ·   T*: frozen text tower, trainable contexts',17,GRAY)
    text(30,1378,'OFF: quality text CE, negative push.   ON: Soft-SROCC.   PI/PT BN statistics frozen downstream; visual weights fine-tuned.',17,PURPLE)
    out.append('</svg>')
    return '\n'.join(out)

if __name__=='__main__':
    target=Path(__file__).resolve().parents[1]/'docs/figures/current_baseline_fused_z_clean.svg'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(render())
    print(target)
