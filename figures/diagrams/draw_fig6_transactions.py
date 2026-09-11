"""Draw Figure 6: CacheFlex transaction paths and source-location cases.

Usage: python3 draw_fig6_transactions.py OUTPUT.pdf
Also writes a PNG preview beside the PDF. Requires Matplotlib and uses its
bundled DejaVu Sans font.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 draw_fig6_transactions.py OUTPUT.pdf")
OUT = Path(sys.argv[1])
if OUT.suffix.lower() != ".pdf":
    raise SystemExit("Output path must end in .pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
W, H = 252, 434
fig = plt.figure(figsize=(W/72, H/72))
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
INK='#292929'; GRAY='#f3f3f3'; SPM='#fff2cc'; ORANGE='#c66b00'; BLUE='#356bb0'

def text(x,y,s,fs=6.1,bold=False,color=INK,ha='center',va='center',**kw):
    return ax.text(x,y,s,fontsize=fs,fontweight='bold' if bold else 'normal',
                   color=color,ha=ha,va=va,linespacing=1.08,**kw)

def box(x,y,w,h,label='',fc='white',edge=INK,fs=6.5,bold=False,rounding=False):
    if rounding:
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',
                            linewidth=.65,edgecolor=edge,facecolor=fc)
    else: patch=Rectangle((x,y),w,h,linewidth=.55,edgecolor=edge,facecolor=fc)
    ax.add_patch(patch)
    if label: text(x+w/2,y+h/2,label,fs,bold)

def arrow(x1,y1,x2,y2,color=INK,dashed=False,both=False,lw=.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='<->' if both else '->',
        mutation_scale=6.3,linewidth=lw,color=color,linestyle=(0,(3,2.3)) if dashed else '-',
        shrinkA=0,shrinkB=0))

def elbow(points,color=ORANGE,lw=1.05):
    path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='->',mutation_scale=6.3,
                                linewidth=lw,color=color,joinstyle='round'))

def cache(x,y,hit=False,local=False):
    # Consistent cache / SPM geometry in all five panels.
    box(x,y,113,60,rounding=True)
    if local: text(x+56.5,y+5.5,'Local L2 install',5.9,True,ORANGE)
    top=y+(12 if local else 7)
    for k,label,fill in [(0,'Cache ways',GRAY),(1,'SPM ways',SPM)]:
        bx=x+4+54*k
        box(bx,top,51,33,fc=fill,fs=6)
        text(bx+25.5,top+5.7,label,6.0)
        for j in range(3):
            box(bx+6,top+12+j*6,39,6,fc=SPM if (k==1 and j==1) else GRAY)
        if k==1:text(bx+25.5,top+21,'Slot N',5.9)
        if hit and k==0:text(bx+25.5,top+21,'Line M (hit)',5.6,True,ORANGE)
    text(x+19,y+53.7,'L2 cache',6.1,True)
    box(x+39,y+49,31,9,'MSHR',fc='#eeeeee',fs=5.7)
    box(x+70,y+49,39,9,'Write buffer',fc='#eeeeee',fs=5.4)
    return {'slot':(x+83.5,top+21),'hit':(x+43,top+21),'mshr':(x+54.5,y+49)}

def queues(x,y,inactive=False):
    box(x+5,y,34,10,'LSQ',fc=GRAY,edge='#cccccc' if inactive else INK,fs=6.4,bold=True)
    box(x+76,y,36,10,'SPM LSQ',fc=GRAY,fs=6.25,bold=True)
    if inactive:
        # Existing faded regular path denotes the path bypassed by pure SPM accesses.
        arrow(x+22,y+10,x+22,y+25,color='#b6b6b6',dashed=True,lw=.65)
    else: arrow(x+39,y+4,x+76,y+4,dashed=True)

def caption(x,y,s):text(x,y,s,7.0,True)

# (a) and (b): keep all original instruction / operand information.
xa, xb = 3, 133
text(xa+56.5,6.5,'SPM_LD {N}, #reg',6.6,True)
text(xa+56.5,15.5,'SPM_ST #reg, {N}',6.6,True)
text(xb+56.5,10.5,'SPM_WB {N}, [M]',6.6,True)
queues(xa,29,inactive=True)
box(xa+5,54,34,11,'L1D',fc=GRAY,edge='#cccccc',fs=6.3,bold=True,rounding=True)
arrow(xa+22,65,xa+22,80,color='#b6b6b6',dashed=True,lw=.65)
cache(xa,80)
arrow(xa+94,39,xa+94,80,color=ORANGE,both=True,lw=1.05)
text(xa+72,59,'LD / ST\naccess',6.1,True,ORANGE)
caption(xa+56.5,148,'(a) SPM_LD / SPM_ST')

queues(xb,29)
arrow(xb+22,18,xb+22,29)
text(xb+57.5,23,'ordering',5.6)
arrow(xb+39,38,xb+76,38,color=BLUE,dashed=True)
text(xb+57.5,44.5,'ACK',5.9,True,BLUE)
box(xb+5,55,34,11,'L1D',fc=GRAY,fs=6.3,bold=True,rounding=True)
arrow(xb+22,39,xb+22,55,color=ORANGE,lw=1.05)
arrow(xb+22,66,xb+22,80,color=ORANGE,lw=1.05)
cache(xb,80)
arrow(xb+94,39,xb+94,80,dashed=True)
text(xb+97.5,62,'read\nreq.',5.8,ha='left')
arrow(xb+80,88,xb+39,40,color=ORANGE,lw=1.05)
text(xb+57,55,'handoff',5.7,True,ORANGE,ha='left')
text(xb+4,73,'coherent write',5.4,True,ORANGE,ha='left',bbox={'facecolor':'white','edgecolor':'none','pad':.4})
caption(xb+56.5,148,'(b) SPM_WB')

text(126,165,'SPM_CP [M], {N}: source-location cases',6.8,True)

def cp_top(x,y,case):
    queues(x,y)
    arrow(x+22,y-6,x+22,y)
    arrow(x+22,y+10,x+22,y+19)
    box(x+4,y+19,46,11, 'L1D (hit)' if case=='c' else ('L1D (miss)' if case=='d' else 'L1D'),
        fc=GRAY,fs=5.8,bold=True,rounding=True)

# (c) L1 hit, (d) L2 hit. Identical scale and aligned captions.
yc=185; cy=226
cp_top(xa,yc,'c'); cc=cache(xa,cy)
# Component-level copy/install interaction; the LSQ retains ordering and ACK.
elbow([(xa+50,yc+24),(xa+86,yc+24),(xa+86,cy)])
text(xa+82,yc+34,'copy / install',5.7,True,ORANGE,ha='right')
arrow(xa+103,cy,xa+103,yc+10,color=BLUE,dashed=True)
text(xa+106,yc+25,'ACK',5.5,True,BLUE,ha='left')
caption(xa+56.5,294,'(c) L1 hit')

cp_top(xb,yc,'d'); dd=cache(xb,cy,hit=True,local=True)
arrow(xb+22,yc+30,xb+22,cy,dashed=True)
text(xb+55,yc+35,'No L1 fill',5.8)
arrow(xb+103,cy,xb+103,yc+10,color=BLUE,dashed=True)
text(xb+106,yc+25,'ACK',5.5,True,BLUE,ha='left')
arrow(xb+51,dd['hit'][1],dd['slot'][0]-14,dd['slot'][1],color=ORANGE,lw=1.05)
caption(xb+56.5,294,'(d) L2 hit')

# (e) centered on its actual cache box; shared key below all five panels.
xe=69.5; ye=312; ey=353
cp_top(xe,ye,'e'); ee=cache(xe,ey)
arrow(xe+22,ye+30,xe+22,ey,dashed=True)
text(xe+56.5,ye+35,'No L1/L2 fill',5.8)
arrow(xe+103,ey,xe+103,ye+10,color=BLUE,dashed=True)
text(xe+111,ye+25,'ACK',5.7,True,BLUE)
arrow(xe+44,ey+32,ee['mshr'][0],ee['mshr'][1],dashed=True)
arrow(ee['mshr'][0]+5,ee['mshr'][1],ee['slot'][0]-14,ee['slot'][1],color=ORANGE,lw=1.05)
text(xe+94,ey+45,'install',5.5,True,ORANGE)
# The lower memory level is outside the L2 boundary. Its interaction enters
# the left side of the MSHR, separately from the internal miss request.
box(8,ey+36,49,15,'LLC / DRAM',fc=GRAY,fs=6.0,bold=True,rounding=True)
elbow([(57,ey+43.5),(xe+42.5,ey+43.5),(xe+42.5,ee['mshr'][1])])
caption(xe+56.5,421,'(e) Full miss')

# Add the compact key on a dedicated strip, extending the page without shrinking panels.
KEY_H=37
ax.set_ylim(H+KEY_H,0); fig.set_size_inches(W/72,(H+KEY_H)/72)
ax.add_patch(Rectangle((2,432),248,35,facecolor='#f6f6f6',edgecolor='none'))
for x,lab,col,dash in [(6,'Request / ordering',INK,True),(94,'Access / update',ORANGE,False),(174,'Completion ACK',BLUE,True)]:
    arrow(x,442,x+13,442,color=col,dashed=dash,lw=.85)
    text(x+17,442,lab,5.4,bold=True,color=col,ha='left')
text(126,453,'{N}: SPM address    [M]: regular address    #reg: register',5.55)
text(126,462,'Gray regular path: bypassed by pure SPM accesses',5.4,color='#555555')

fig.savefig(OUT,transparent=False)
fig.savefig(OUT.with_suffix('.png'),dpi=220)
plt.close(fig)
