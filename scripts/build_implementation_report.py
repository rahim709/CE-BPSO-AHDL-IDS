"""Export the consolidated implementation report to an editable Word document.

Optional export dependency: python-docx. The ML runtime does not require it.
"""
from __future__ import annotations
import argparse
from datetime import date
import re
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

SOURCE = Path('docs/reports/CE-BPSO-AHDL-IDS-Implementation-Report.md')


def add_inline(paragraph, text: str) -> None:
    pieces = re.split(r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))', text)
    for piece in pieces:
        if not piece:
            continue
        link = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', piece)
        if link:
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'), paragraph.part.relate_to(link[2], RT.HYPERLINK, is_external=True))
            run = OxmlElement('w:r')
            props = OxmlElement('w:rPr')
            color = OxmlElement('w:color')
            color.set(qn('w:val'), '244766')
            props.append(color)
            run.append(props)
            label = OxmlElement('w:t')
            label.text = link[1]
            run.append(label)
            hyperlink.append(run)
            paragraph._p.append(hyperlink)
        elif piece.startswith('**') and piece.endswith('**'):
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith('`') and piece.endswith('`'):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(9)
        else:
            paragraph.add_run(piece)


def make_table(document, lines: list[str]) -> None:
    rows = [[v.strip() for v in line.strip().strip('|').split('|')] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r':?-+:?', value.replace(' ', '')) for value in r)]
    if not rows:
        return
    table = document.add_table(rows=0, cols=len(rows[0]))
    table.style = 'Table Grid'
    table.autofit = True
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(3)
            add_inline(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(8.5 if len(rows[0]) >= 5 else 9)
                if i == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor.from_string('FFFFFF')
            if i == 0:
                shading = OxmlElement('w:shd')
                shading.set(qn('w:fill'), '244766')
                cell._tc.get_or_add_tcPr().append(shading)
        props = table.rows[-1]._tr.get_or_add_trPr()
        no_split = OxmlElement('w:cantSplit')
        props.append(no_split)
        if i == 0:
            repeat = OxmlElement('w:tblHeader')
            props.append(repeat)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.source
    output = args.output or source.with_suffix('.docx')
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(.7)
    section.bottom_margin = Inches(.65)
    section.left_margin = Inches(.7)
    section.right_margin = Inches(.7)
    normal = document.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.05
    for level, size in [(1, 16), (2, 12), (3, 11)]:
        style = document.styles[f'Heading {level}']
        style.font.name = 'Calibri'
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string('244766')
        style.paragraph_format.keep_with_next = True
    header = section.header.paragraphs[0]
    title = source.read_text().splitlines()[0].removeprefix('# ')
    header.add_run('CE-BPSO-AHDL-IDS | Project Research').font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run(f'{date.today().isoformat()} | Page ').font.size = Pt(8)
    page = OxmlElement('w:fldSimple')
    page.set(qn('w:instr'), 'PAGE')
    footer._p.append(page)
    document.core_properties.title = title
    document.core_properties.subject = 'Implemented framework, reproducibility, measured experiments and limitations'
    document.core_properties.author = 'Abdur Rahim; Arpit Kumar Singh; Gosala Buela'
    document.core_properties.keywords = 'BPSO, CNN, RNN, intrusion detection, reproducibility, synthetic data'

    lines = source.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith('```'):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith('```'):
                code.append(lines[index])
                index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.line_spacing = 1
            paragraph.paragraph_format.space_after = Pt(8)
            run = paragraph.add_run('\n'.join(code))
            run.font.name = 'Consolas'
            run.font.size = Pt(8)
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'), 'F2F4F6')
            paragraph._p.get_or_add_pPr().append(shade)
            index += 1
            continue
        if line.startswith('|'):
            table_lines = []
            while index < len(lines) and lines[index].startswith('|'):
                table_lines.append(lines[index])
                index += 1
            make_table(document, table_lines)
            continue
        picture = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if picture:
            document.add_picture(str(source.parent / picture[2]), width=Inches(5.8))
            caption = document.add_paragraph(picture[1])
            caption.alignment = 1
            caption.paragraph_format.space_after = Pt(8)
            for run in caption.runs:
                run.font.size = Pt(9)
                run.italic = True
        elif line.startswith('# '):
            document.add_heading(line[2:], 0)
        elif line.startswith('## '):
            document.add_heading(line[3:], 1)
        elif line.startswith('### '):
            document.add_heading(line[4:], 2)
        elif line.startswith('- '):
            add_inline(document.add_paragraph(style='List Bullet'), line[2:])
        elif re.match(r'^\d+\. ', line):
            add_inline(document.add_paragraph(style='List Number'), re.sub(r'^\d+\. ', '', line))
        else:
            paragraph_lines = [line]
            while index + 1 < len(lines) and lines[index+1].strip() and not re.match(r'^(#|\||```|- |\d+\. )', lines[index+1]):
                index += 1
                paragraph_lines.append(lines[index])
            add_inline(document.add_paragraph(), ' '.join(paragraph_lines))
        index += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    # Check the generated package can be read and retains substantive content.
    reread = Document(output)
    headings = [p.text for p in reread.paragraphs if p.style.name == 'Heading 1']
    expected_headings = sum(line.startswith('## ') for line in lines)
    if len(headings) != expected_headings or not any(p.text == title for p in reread.paragraphs):
        raise RuntimeError('Incomplete Word export')
    print(f'Created {output}; {len(headings)} sections, {len(reread.tables)} tables, {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
