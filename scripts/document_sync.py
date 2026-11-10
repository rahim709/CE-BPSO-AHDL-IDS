"""Export current LaTeX reports to Markdown; no model runs or evidence rewrites."""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / 'docs/latex'


def group(s, pos):
    assert s[pos] == '{'
    depth, end = 1, pos + 1
    while depth:
        if s[end] == '{' and s[end-1] != '\\': depth += 1
        if s[end] == '}' and s[end-1] != '\\': depth -= 1
        end += 1
    return s[pos+1:end-1], end


def command(s, name, render):
    pattern = re.compile(r'\\' + name + r'\s*\{')
    out, pos = [], 0
    while match := pattern.search(s, pos):
        out.append(s[pos:match.start()])
        value, pos = group(s, match.end()-1)
        out.append(render(value))
    return ''.join(out) + s[pos:]


def expand(path, stack=()):
    assert path not in stack
    def include(match):
        child = LATEX / match[1]
        if not child.suffix: child = child.with_suffix('.tex')
        return expand(child, (*stack, path))
    return re.sub(r'\\input\{([^}]+)\}', include, path.read_text())


def roman(n):
    out = ''
    for value, token in [(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while n >= value: out += token; n -= value
    return out


class Exporter:
    def __init__(self, source, report):
        self.report, self.tokens, self.refs = report, {}, {}
        self.keys = list(dict.fromkeys(k for c in re.findall(r'\\cite\{([^}]+)\}', source) for k in c.split(',')))
        self.cites = {k:str(i+1) for i,k in enumerate(self.keys)}
        self.macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^{}]*)\}', source))
        self.figures, self.tables = [], []
        ch = sec = sub = fig = tab = 0
        current = ''
        pattern = r'\\(chapter|section|subsection)\{[^}]*\}|\\begin\{(figure|table|longtable)\}|\\label\{([^}]+)\}'
        for m in re.finditer(pattern, source):
            heading, env, label = m.groups()
            if heading == 'chapter': ch += 1; sec = sub = fig = tab = 0; current = str(ch)
            elif heading == 'section': sec += 1; sub = 0; current = f'{ch}.{sec}' if report else roman(sec)
            elif heading == 'subsection': sub += 1; current = f'{ch}.{sec}.{sub}' if report else f'{roman(sec)}-{chr(64+sub)}'
            elif env:
                if env == 'figure': fig += 1; n = fig; queue = self.figures
                else: tab += 1; n = tab; queue = self.tables
                current = f'{ch}.{n}' if report else str(n)
                queue.append(current)
            elif label: self.refs[label] = current
        self.ch = self.sec = self.sub = 0
        self.appendix = False

    def protect(self, value):
        key = f'@@DOCBLOCK{len(self.tokens)}@@'; self.tokens[key] = value; return key

    def inline(self, s):
        s = command(s, 'ref', lambda v:self.refs[v])
        s = command(s, 'cite', lambda v:'['+', '.join(self.cites[k] for k in v.split(','))+']')
        for name,value in self.macros.items(): s = s.replace('\\'+name+'{}', value)
        s = s.replace(r'\macroF',r'\mathrm{F1}_{\mathrm{macro}}')
        s = s.replace(r'$\pm$', '±')
        for name in ['file','path','texttt']:
            s = command(s,name,lambda v:self.protect('`'+v.replace(r'\_', '_').replace(r'\%', '%')+'`'))
        for name,a,b in [('mbox','',''),('emph','*','*'),('textbf','**','**'),('fbox','','')]:
            s = command(s,name,lambda v,a=a,b=b:a+self.inline(v)+b)
        return s.replace(r'\%','%').replace(r'\_','_').replace(r'\&','&').replace('~',' ').replace('---','—').replace('--','–')

    def caption(self, block):
        m = re.search(r'\\caption\s*\{',block)
        return self.inline(group(block,m.end()-1)[0]) if m else ''

    def figure(self, block):
        number = self.figures.pop(0); caption = self.caption(block)
        m = re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',block)
        if m:
            asset = LATEX/'figures'/m[1]
            if not asset.exists(): asset = LATEX/m[1]
            assert asset.is_file(),asset
            png = asset.with_suffix('.png')
            if not png.exists() and shutil.which('pdftoppm'):
                subprocess.run(['pdftoppm','-singlefile','-scale-to','1600','-png',str(asset),str(png.with_suffix(''))],check=True,capture_output=True)
            chosen = png if png.exists() else asset
            result = f'![Figure {number}. {caption}](../latex/{chosen.relative_to(LATEX)})'
        else:
            pdf = 'implementation_report.pdf' if self.report else 'manuscript.pdf'
            result = f'**Figure {number}. {caption}** See the [framework diagram](../latex/{pdf}).'
        return self.protect('\n\n'+result+'\n\n')

    def table(self, block):
        number = self.tables.pop(0); caption = self.caption(block)
        m = re.search(r'\\begin\{(?:tabular|longtable)\}',block)
        assert m,block[:80]
        _,start = group(block,block.index('{',m.end()))
        end = re.search(r'\\end\{(?:tabular|longtable)\}',block[start:]).start()+start
        content = re.sub(r'\\(?:toprule|midrule|bottomrule|hline|endhead)','',block[start:end])
        rows = [[' '.join(self.inline(cell).split()) for cell in row.split('&')] for row in re.split(r'\\\\',content) if '&' in row]
        assert rows and all(len(row)==len(rows[0]) for row in rows)
        out = f'**Table {number}. {caption}**\n\n' if caption else ''
        out += '| '+' | '.join(rows[0])+' |\n| '+' | '.join(['---']*len(rows[0]))+' |\n'
        out += '\n'.join('| '+' | '.join(row)+' |' for row in rows[1:])
        return self.protect('\n\n'+out+'\n\n')

    def heading(self,m):
        kind,title = m.groups()
        if kind=='chapter':
            self.ch += 1; self.sec = self.sub = 0
            n = chr(64+self.ch) if self.appendix else str(self.ch); level = '##'
        elif kind=='section':
            self.sec += 1; self.sub = 0
            prefix = chr(64+self.ch) if self.appendix else str(self.ch)
            n = f'{prefix}.{self.sec}' if self.report else str(self.sec); level = '###' if self.report else '##'
        else:
            self.sub += 1; prefix = chr(64+self.ch) if self.appendix else str(self.ch)
            n = f'{prefix}.{self.sec}.{self.sub}' if self.report else f'{self.sec}.{self.sub}'; level = '####' if self.report else '###'
        return '\n\n'+level+' '+n+'. '+self.inline(title)+'\n\n'

    def convert(self,body):
        # Keep prose between aligned equations outside the Markdown math blocks.
        body = command(body, 'intertext', lambda value:r'\end{align}'+'\n'+value+'\n'+r'\begin{align}')
        body = re.sub(r'(?<!\\)%[^\n]*','',body)
        body = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}',lambda m:self.protect('\n\n```bash\n'+m[0].split('}',1)[1].rsplit(r'\end',1)[0].strip()+'\n```\n\n'),body,flags=re.S)
        def quote(m):
            value=re.sub(r'\\(?:begin|end)\{quote\}|\\(?:ttfamily|sloppy)\b','',m[0]).replace('{','').replace('}','').strip()
            return self.protect('\n\n```text\n'+value+'\n```\n\n')
        body=re.sub(r'\\begin\{quote\}.*?\\end\{quote\}',quote,body,flags=re.S)
        body = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',lambda m:self.figure(m[0]),body,flags=re.S)
        body = re.sub(r'\\begin\{table\}.*?\\end\{table\}',lambda m:self.table(m[0]),body,flags=re.S)
        body = re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',lambda m:self.table(m[0]),body,flags=re.S)
        def math(m):
            value = re.sub(r'^\\begin\{(?:equation|align)\}|\\end\{(?:equation|align)\}$','',m[0])
            value = value.replace(r'\begin{split}',r'\begin{aligned}').replace(r'\end{split}',r'\end{aligned}').replace(r'\macroF',r'\mathrm{F1}_{\mathrm{macro}}')
            if m[0].startswith(r'\begin{align}'): value = '\\begin{aligned}\n'+value+'\n\\end{aligned}'
            return self.protect('\n\n$$\n'+value.strip()+'\n$$\n\n')
        body = re.sub(r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}',math,body,flags=re.S)
        body = self.inline(body)
        body = re.sub(r'(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)',lambda m:self.protect(m[0]),body,flags=re.S)
        body = re.sub(r'\\begin\{minipage\}\{[^}]*\}|\\end\{minipage\}','',body)
        body = re.sub(r'\\label\{[^}]*\}','',body)
        parts = body.split(r'\appendix')
        for i,part in enumerate(parts):
            if i:self.appendix=True;self.ch=0
            parts[i]=re.sub(r'\\(chapter|section|subsection)\{([^}]+)\}',self.heading,part)
        body='\n'.join(parts)
        body=body.replace(r'\begin{IEEEkeywords}','\n\n**Keywords:** ').replace(r'\end{IEEEkeywords}','\n\n')
        body=re.sub(r'\\(?:begin|end)\{abstract\}','\n\n',body)
        body=body.replace(r'\begin{enumerate}','\n').replace(r'\end{enumerate}','\n').replace(r'\item','\n- ')
        body=re.sub(r'\\(?:FloatBarrier|tableofcontents|listoffigures|listoftables|centering|small|noindent)\b','',body)
        body=re.sub(r'\\par\b','\n',body)
        body=re.sub(r'\{([^{}]*)\}',r'\1',body)
        body=re.sub(r'\n[ \t]+','\n',body)
        for key,value in reversed(list(self.tokens.items())):body=body.replace(key,value)
        assert not self.figures and not self.tables
        return re.sub(r'\n{3,}','\n\n',body).strip()+'\n'

    def references(self):
        bib=(LATEX/'references.bib').read_text();entries={}
        for m in re.finditer(r'@\w+\{([^,]+),',bib):
            block,_=group(bib,bib.index('{',m.start()));fields={}
            for f in re.finditer(r'(\w+)\s*=\s*\{',block):fields[f[1]]=group(block,f.end()-1)[0]
            entries[m[1]]=fields
        out=['\n## References\n']
        clean=lambda s:s.replace('{','').replace('}','').replace('---','—').replace('--','–')
        for key in self.keys:
            d=entries[key];parts=[clean(d['author']).replace(' and ','; '),'“'+clean(d['title'])+'”']
            for f in ['journal','booktitle','institution','volume','number','pages','year','note']:
                if f in d:parts.append({'volume':'vol. ','number':'no. ','pages':'pp. '}.get(f,'')+clean(d[f]))
            if 'doi' in d:parts.append('[DOI](https://doi.org/'+d['doi']+')')
            if 'url' in d:parts.append('[Source]('+d['url']+')')
            out.append('['+self.cites[key]+'] '+'. '.join(parts)+'.\n')
        return '\n'.join(out)


def main():
    title=re.search(r'\\title\{([^}]+)\}',expand(LATEX/'manuscript.tex'))[1]
    for report,source,output in [(False,'manuscript.tex','CE-BPSO-AHDL-IDS.md'),(True,'implementation_report.tex','CE-BPSO-AHDL-IDS-Implementation-Report.md')]:
        text=expand(LATEX/source);export=Exporter(text,report)
        body=text[text.index(r'\begin{abstract}'):text.index(r'\bibliographystyle')]
        suffix=': Implementation and Experimental Evaluation Report' if report else ''
        metadata='**Authors:** Abdur Rahim, Arpit Kumar Singh, Gosala Buela\n\n**Affiliation:** IIITDM Kancheepuram, Chennai, India\n\n**Experimental revision:** 8 November 2026\n\n**Editorial synchronization:** 10 November 2026\n\n## Abstract\n\n'
        result='# '+title+suffix+'\n\n'+metadata+export.convert(body)+export.references()
        assert '@@DOCBLOCK' not in result
        (ROOT/'docs/reports'/output).write_text(result)
        print('Synchronized',output,'from',source)


if __name__=='__main__':main()
