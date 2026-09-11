"""Draw Figure 5: dual-LSQ coordination for CacheFlex bridge instructions.

Usage: python3 draw_fig5_lsq.py OUTPUT.pdf
Also writes a PNG preview beside the PDF. Requires PyMuPDF and Matplotlib;
the embedded DejaVu Sans fonts are supplied by Matplotlib.
"""

from pathlib import Path
import math
import sys

import fitz
from matplotlib import get_data_path


if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 draw_fig5_lsq.py OUTPUT.pdf")
OUT = Path(sys.argv[1])
if OUT.suffix.lower() != ".pdf":
    raise SystemExit("Output path must end in .pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 1300, 580
FONT_DIR = Path(get_data_path()) / 'fonts/ttf'
FONTS = {
    'regular': fitz.Font(fontfile=str(FONT_DIR / 'DejaVuSans.ttf')),
    'bold': fitz.Font(fontfile=str(FONT_DIR / 'DejaVuSans-Bold.ttf')),
}
DOC = fitz.open()
PAGE = DOC.new_page(width=W, height=H)
for name, filename in [('regular', 'DejaVuSans.ttf'), ('bold', 'DejaVuSans-Bold.ttf')]:
    PAGE.insert_font(fontname=name, fontfile=str(FONT_DIR / filename))


def rgb(hexcode):
    return tuple(int(hexcode[i:i+2], 16) / 255 for i in (0, 2, 4))


INK = rgb('17212B')
GRID = rgb('66717A')
BLUE = rgb('0072B2')
RED = rgb('D5524A')
REGULAR = rgb('F4F6F8')
SPM = rgb('FFF3D4')
HEADER = rgb('E6EBEF')


def text(x, y, value, size=17.5, bold=False, color=INK, leading=19.0):
    """Text centered on a supplied point, with explicit line spacing."""
    name = 'bold' if bold else 'regular'
    font = FONTS[name]
    lines = value.split('\n')
    # fitz y coordinates designate the text baseline.
    for n, line in enumerate(lines):
        cy = y + (n - (len(lines) - 1) / 2) * leading
        baseline = cy + size * (font.ascender + font.descender) / 2
        left = x - font.text_length(line, fontsize=size) / 2
        PAGE.insert_text((left, baseline), line, fontsize=size,
                         fontname=name, color=color)


SQ_WIDTHS = [40, 75, 52, 74, 86]
LQ_WIDTHS = [40, 75, 52, 90]
ROW_H = 40


def table(x, y, columns, rows, title, spm=False, highlights=None):
    widths = SQ_WIDTHS if len(columns) == 5 else LQ_WIDTHS
    width, height = sum(widths), ROW_H * (len(rows) + 1)
    PAGE.draw_rect(fitz.Rect(x, y, x+width, y+height),
                   color=GRID, fill=SPM if spm else REGULAR, width=0.9)
    PAGE.draw_rect(fitz.Rect(x, y, x+width, y+ROW_H),
                   color=None, fill=HEADER, overlay=True)
    for i in range(1, len(rows) + 1):
        PAGE.draw_line((x, y + i*ROW_H), (x+width, y+i*ROW_H), color=GRID, width=0.8)
    xx = x
    for col, cw in zip(columns, widths):
        if xx > x:
            PAGE.draw_line((xx, y), (xx, y+height), color=GRID, width=0.8)
        for line in col.split('\n'):
            assert FONTS['bold'].text_length(line, fontsize=17.5) <= cw - 2, (col, cw)
        text(xx+cw/2, y+ROW_H/2, col, bold=True)
        xx += cw
    for n, row in enumerate(rows):
        xx = x
        for value, cw in zip(row, widths):
            # Assert that no transcribed text can overflow a cell.
            for line in value.split('\n'):
                assert FONTS['regular'].text_length(line, fontsize=17.5) <= cw - 3, (value, cw)
            text(xx+cw/2, y+(n+1.5)*ROW_H, value)
            xx += cw
    for index, color in (highlights or {}).items():
        PAGE.draw_rect(fitz.Rect(x, y+(index+1)*ROW_H,
                                x+width, y+(index+2)*ROW_H),
                       color=color, width=2.1)
    text(x+width/2, y-16, title, size=20, bold=True)
    return (x, x+width, [y+(n+1.5)*ROW_H for n in range(len(rows))])


def arrow(start, end, dashed=False):
    PAGE.draw_line(start, end, color=INK, width=1.8,
                   dashes='[6 4] 0' if dashed else None)
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    length, spread = 8.2, 3.6
    rear = (end[0]-length*math.cos(angle), end[1]-length*math.sin(angle))
    side1 = (rear[0]+spread*math.sin(angle), rear[1]-spread*math.cos(angle))
    side2 = (rear[0]-spread*math.sin(angle), rear[1]+spread*math.cos(angle))
    shape = PAGE.new_shape()
    shape.draw_polyline([end, side1, side2, end])
    shape.finish(color=INK, fill=INK, width=0.7, closePath=True)
    shape.commit()


ELL5, ELL4 = ['…']*5, ['…']*4
BLANK5, BLANK4 = ['']*5, ['']*4
SQ_COLS = ['Seq', 'Op', 'ST\nAddr', 'ST\nData', 'Note']
LQ_COLS = ['Seq', 'Op', 'LD\nAddr', 'Note']


def panel(x, y, which):
    sq_x = x
    lq_x = x + sum(SQ_WIDTHS) + 50
    cx = x + sum(SQ_WIDTHS) + 25
    if which == 'a':
        sq = table(sq_x, y, SQ_COLS, [
            ['1', 'ST', 'A', 'xxx', 'older ST'],
            ['2', 'ST', 'B', 'xxx', 'no-alias'],
            ['4', 'spm_cp', 'A', 'a (spm\naddr)', 'bridge\nentry'], ELL5],
            'Regular SQ', highlights={0: BLUE, 2: RED})
        lq = table(lq_x, y, LQ_COLS, [
            ['7', 'LD', 'A', 'younger\nLD'],
            ['8', 'LD', 'C', 'no-alias'], BLANK4, ELL4],
            'Regular LQ', highlights={0: BLUE})
        arrow((sq[1]+2, sq[2][0]), (lq[0]-3, lq[2][0]))
        arrow((sq[1]+2, sq[2][2]), (lq[0]-3, lq[2][0]+3), dashed=True)
        arrow((x+42, y+219), (x+73, y+219))
        text(x+134, y+219, 'forwarding')
        arrow((x+226, y+219), (x+257, y+219), dashed=True)
        text(x+404, y+219, 'no forwarding, no squash')
        subtitle = '(a) spm_cp on the regular side'
    elif which == 'b':
        sq = table(sq_x, y, SQ_COLS, [
            ['4', 'spm_cp', 'a', 'xxx', 'install\npending'],
            ['6', 'spm_st', 'b', 'xxx', 'no-alias'], BLANK5, ELL5],
            'SPM SQ', spm=True, highlights={0: RED})
        lq = table(lq_x, y, LQ_COLS, [
            ['3', 'spm_ld', 'a', 'older\nload'],
            ['5', 'spm_ld', 'a', 'younger\nload'], BLANK4, ELL4],
            'SPM LQ', spm=True, highlights={1: BLUE})
        arrow((sq[1]+2, sq[2][0]), (lq[0]-3, lq[2][1]))
        text(x+317, y+219, 'RAW hazard: hold/squash on conflict')
        subtitle = '(b) spm_cp on the SPM side'
    elif which == 'c':
        sq = table(sq_x, y, SQ_COLS, [
            ['1', 'ST', 'A', 'xxx', 'older ST'],
            ['2', 'ST', 'B', 'xxx', 'no-alias'],
            ['4', 'spm_wb', 'A', 'a (spm\naddr)', 'bridge\nentry'], ELL5],
            'Regular SQ', highlights={2: RED})
        lq = table(lq_x, y, LQ_COLS, [
            ['5', 'LD', 'A', 'younger\nLD'],
            ['9', 'LD', 'C', 'no-alias'], BLANK4, ELL4],
            'Regular LQ', highlights={0: BLUE})
        arrow((sq[1]+2, sq[2][2]), (lq[0]-3, lq[2][0]))
        text(x+317, y+219, 'RAW: wait until coherent install (no forwarding)')
        subtitle = '(c) spm_wb on the regular side'
    else:
        sq = table(sq_x, y, SQ_COLS, [
            ['3', 'spm_st', 'a', 'xxx', 'older ST'],
            ['7', 'spm_st', 'a', 'xxx', 'younger\nST'], BLANK5, ELL5],
            'SPM SQ', spm=True, highlights={1: BLUE})
        lq = table(lq_x, y, LQ_COLS, [
            ['4', 'spm_wb', 'a', 'wb\npending'],
            ['8', 'spm_ld', 'd', 'no-alias'], BLANK4, ELL4],
            'SPM LQ', spm=True, highlights={0: RED})
        arrow((lq[0]-2, lq[2][0]), (sq[1]+3, sq[2][1]))
        text(x+317, y+219, 'WAR hazard: stall on conflict')
        subtitle = '(d) spm_wb on the SPM side'
    text(x+317, y+245, subtitle, size=18)


panel(6, 30, 'a')
panel(660, 30, 'b')
panel(6, 319, 'c')
panel(660, 319, 'd')
DOC.set_metadata({'title': 'CacheFlex: dual-LSQ coordination for bridge instructions',
                  'author': '', 'subject': 'Figure 5; vector artwork with embedded DejaVu Sans fonts'})
DOC.subset_fonts()
DOC.save(OUT, garbage=4, deflate=True)
PAGE.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT.with_suffix('.png'))
print(OUT)
print('Fonts:', PAGE.get_fonts())
