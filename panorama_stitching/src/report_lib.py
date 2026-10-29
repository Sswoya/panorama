from __future__ import annotations

import os

import cv2
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SONG = "宋体"
HEI = "黑体"
KAI = "楷体"
MONO = "Consolas"


# 基础排版
def set_run_font(run, east=SONG, ascii_font="Times New Roman", size=12, bold=False, color=None):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east)
    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)


def shade(element, fill: str):
    pPr = element.get_or_add_pPr() if hasattr(element, "get_or_add_pPr") else element
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def para_border(p, color="D9D9D9", size=6):
    pPr = p._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "4")
        el.set(qn("w:color"), color)
        borders.append(el)
    pPr.append(borders)


def body(doc, text: str, indent=True, size=12, line=1.5, space_after=3, bold=False, align=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # 支持用 **文字** 标记行内加粗
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        run = p.add_run(seg)
        set_run_font(run, SONG, "Times New Roman", size, bold=(bold or i % 2 == 1))
    return p


def rich_body(doc, parts, indent=True, size=12, line=1.5, space_after=3):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for text, opt in parts:
        run = p.add_run(text)
        if opt.get("mono"):
            set_run_font(run, MONO, MONO, size - 1)
            run.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        else:
            set_run_font(run, SONG, "Times New Roman", size, bold=opt.get("bold", False))
    return p


def heading(doc, text: str, level: int = 1, page_break: bool = False):
    sizes = {1: 16, 2: 14, 3: 12}
    before = {1: 18, 2: 12, 3: 9}
    after = {1: 10, 2: 8, 3: 6}
    p = doc.add_heading(level=level)
    pf = p.paragraph_format
    pf.space_before = Pt(before[level])
    pf.space_after = Pt(after[level])
    pf.line_spacing = 1.3
    pf.first_line_indent = Pt(0)
    pf.page_break_before = page_break
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_run_font(run, HEI, "Times New Roman", sizes[level], bold=(level >= 2))
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    # 让标题进入导航窗格/自动目录
    p.style = doc.styles[f"Heading {level}"]
    return p


def caption(doc, text: str, kind: str = "图"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(10 if kind == "图" else 4)
    pf.line_spacing = 1.2
    # 表题的下一段是表格，绑在一起，避免"表题独自留在页底"
    pf.keep_with_next = kind == "表"
    run = p.add_run(text)
    set_run_font(run, SONG, "Times New Roman", 10.5, bold=True)
    return p


def image(doc, path: str, width_cm: float = 15.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    # 图和它的图题必须同页
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    return p


def figure(doc, path: str, cap: str, width_cm: float = 15.0):
    image(doc, path, width_cm)
    caption(doc, cap, "图")


def table(doc, headers, rows, cap: str, widths=None, font_size=9.5, header_fill="E8EEF7"):
    caption(doc, cap, "表")
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        run = p.add_run(str(h))
        set_run_font(run, HEI, "Times New Roman", font_size, bold=True)
        shade(hdr[i]._tc.get_or_add_tcPr(), header_fill)

    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(v))
            set_run_font(run, SONG, "Times New Roman", font_size)

    _finalize_table(t, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _finalize_table(t, widths):
    tbl = t._tbl
    tblPr = tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)

    if widths:
        # tblW 用 DXA（1/20 pt）给出总宽，避免百分比布局把列宽约束失效
        total = int(sum(widths) * 567)
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW")
            tblPr.append(tblW)
        tblW.set(qn("w:type"), "dxa")
        tblW.set(qn("w:w"), str(total))
        for i, w in enumerate(widths):
            if i < len(t.columns):
                t.columns[i].width = Cm(w)
            for row in t.rows:
                if i < len(row.cells):
                    row.cells[i].width = Cm(w)

    # 表头行重复
    if len(t.rows) > 0:
        trPr = t.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader")
        hdr.set(qn("w:val"), "true")
        trPr.append(hdr)

    # 所有行不跨页断开
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        trPr.append(cant)

    # 小表格整体不跨页（除最后一行外都设置"与下段同页"）
    if len(t.rows) <= 9:
        for row in t.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True


def code_block(doc, code: str, cap: str | None = None, size=8.5):
    """代码块：等宽字体 + 浅灰底 + 细边框。"""
    lines = code.rstrip("\n").split("\n")
    n = len(lines)
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.left_indent = Pt(6)
        pf.right_indent = Pt(6)
        shade(p._p.get_or_add_pPr(), "F6F7F9")
        if i == n - 1:
            pf.space_after = Pt(4)
        run = p.add_run(line if line else " ")
        set_run_font(run, MONO, MONO, size)
    if cap:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(10)
        run = p.add_run(cap)
        set_run_font(run, SONG, "Times New Roman", 10.5, bold=True)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run_font(run, SONG, "Times New Roman", 9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    run._r.addnext(fld)


def header_text(section, text: str):
    hdr = section.header
    p = hdr.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, SONG, "Times New Roman", 9)
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.6)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
    return sec


def new_page(doc):
    from docx.enum.text import WD_BREAK

    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


# 图片预处理
def prepare_figure(src: str, dst_dir: str, max_width: int = 1500, quality: int = 92) -> str:
    from .io_utils import imread, imwrite  # noqa: F401  (保持与其他模块一致的读写方式)

    img = imread(src)
    if img is None:
        raise FileNotFoundError(src)
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0] + ".png"
    out = os.path.join(dst_dir, name)
    imwrite(out, img)
    return out
