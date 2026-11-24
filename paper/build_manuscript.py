"""Build the manuscript and explanatory figures from saved tables, without model runs.

    python -X utf8 paper/build_manuscript.py

Requires matplotlib, reportlab and Pillow. The editable manuscript is DRAFT_v5.md.
"""
import csv
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUTPUT=PROJECT/'output'/'pdf'


def table(name):
    with (ROOT/'tables'/f'{name}.csv').open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    ink='#162335';blue='#2864a0';teal='#148478';grey='#e1e6ec'
    fig,ax=plt.subplots(figsize=(8.4,2.55),layout='constrained')
    ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    boxes=[(.15,1.15,2.15,1.25,'Codebook\nx -> v'),(3.1,1.15,2.8,1.25,'Rule-token states\nkeys become addressable'),
           (6.7,1.15,2.95,1.25,'Reader query heads\nretrieve P(v)')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',
                                   facecolor='#f2f6fa',edgecolor=blue,linewidth=1.4))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=ink,fontsize=11)
    for left,right,label in ((2.35,3.0,'binding'),(5.98,6.62,'read')):
        ax.annotate('',xy=(right,1.75),xytext=(left,1.75),arrowprops=dict(arrowstyle='->',color=blue,lw=1.8))
        ax.text((left+right)/2,2.5,label,ha='center',fontsize=9,color=ink)
    ax.text(1.2,.68,'Block: rule -> codebook',ha='center',color='#9b3b33',fontsize=9)
    ax.text(4.5,.68,'Restore clean K or V',ha='center',color=teal,fontsize=9)
    ax.text(8.2,.68,'READ is the primary endpoint',ha='center',color=teal,fontsize=9)
    ax.text(5,.12,'APPLY also requires decoding x and composing the two lookups; its recovery is less consistent.',
            ha='center',color=ink,fontsize=9)
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'f10_circuit.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

    rows=[r for r in table('t22_receiver_fresh') if r['Run'].startswith('codes_')]
    names={'codes_qwen3_4b':'Qwen3-4B','codes_qwen3_8b':'Qwen3-8B','codes_phi35_mini':'Phi-3.5',
           'codes_mistral_7b':'Mistral-7B','codes_gemma2_9b':'Gemma-2-9B','codes_llama31_8b':'Llama-3.1-8B'}
    assert len(rows)==6,'The fresh-bundle receiver figure needs all six codes models'
    fig,ax=plt.subplots(figsize=(8.4,3.5),layout='constrained')
    xs=np.arange(len(rows));width=.19
    variants=[('READ blocked','Blocked',grey),('READ K only','K only',blue),('READ V only','V only','#d28a3b'),('READ clean','Clean',teal)]
    for j,(key,label,color) in enumerate(variants):
        ax.bar(xs+(j-1.5)*width,[float(r[key]) for r in rows],width*.95,label=label,color=color,edgecolor='white',linewidth=.5)
    ax.set_ylim(0,1.05);ax.set_ylabel('READ accuracy');ax.set_xticks(xs,[names[r['Run']] for r in rows],fontsize=9)
    ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False)
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'f11_receiver_fresh.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def inline(text):
    text=html.escape(text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#245b87">\1</link>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier" size="8.7">\1</font>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
    return text


def make_pdf():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,PageBreak,Preformatted
    OUTPUT.mkdir(parents=True,exist_ok=True)
    fonts=Path('C:/Windows/Fonts')
    font_files={'Paper':'times.ttf','PaperBold':'timesbd.ttf','PaperItalic':'timesi.ttf','PaperBoldItalic':'timesbi.ttf'}
    for name,file in font_files.items():
        if (fonts/file).exists():pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    if 'Paper' in pdfmetrics.getRegisteredFontNames():
        font='Paper';bold='PaperBold';italic='PaperItalic'
        pdfmetrics.registerFontFamily('Paper',normal='Paper',bold=bold,italic=italic,boldItalic='PaperBoldItalic')
    else:font='Times-Roman';bold='Times-Bold';italic='Times-Italic'
    width=504
    body=ParagraphStyle('body',fontName=font,fontSize=10.6,leading=13.7,alignment=TA_JUSTIFY,spaceAfter=6.3,
                        splitLongWords=True,allowWidows=0,allowOrphans=0)
    styles={
        'body':body,
        'title':ParagraphStyle('title',parent=body,fontName=bold,fontSize=21,leading=24.5,alignment=TA_CENTER,spaceAfter=10),
        'meta':ParagraphStyle('meta',parent=body,fontName=italic,fontSize=9.3,leading=12,alignment=TA_CENTER,spaceAfter=15),
        'h1':ParagraphStyle('h1',parent=body,fontName=bold,fontSize=13.1,leading=16,spaceBefore=12,spaceAfter=7,keepWithNext=True),
        'h2':ParagraphStyle('h2',parent=body,fontName=bold,fontSize=11.2,leading=14,spaceBefore=8,spaceAfter=5,keepWithNext=True),
        'cell':ParagraphStyle('cell',parent=body,fontSize=8.8,leading=10.7,alignment=0,spaceAfter=0),
        'header':ParagraphStyle('header',parent=body,fontName=bold,fontSize=8.7,leading=10.5,alignment=0,spaceAfter=0),
        'caption':ParagraphStyle('caption',parent=body,fontSize=9.2,leading=11.6,alignment=0,spaceAfter=8),
        'ref':ParagraphStyle('ref',parent=body,fontSize=9.3,leading=11.7,alignment=0,spaceAfter=6,leftIndent=18,firstLineIndent=-18),
        'code':ParagraphStyle('code',fontName='Courier',fontSize=8.2,leading=11,spaceBefore=5,spaceAfter=8),
    }
    source=(ROOT/'DRAFT_v5.md').read_text(encoding='utf-8')
    assert not re.search(r'\{\{|<AUTHORS>|TODO|TBD',source),'Unresolved manuscript placeholder'
    lines=source.splitlines();story=[];i=0;references=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['title']));i+=1;continue
        if line.startswith('## '):
            label=line[3:]
            if label=='References' or label.startswith('Appendix A'):story.append(PageBreak())
            references=label=='References'
            story.append(Paragraph(inline(label),styles['h1']));i+=1;continue
        if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['h2']));i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(code),styles['code']));i+=1;continue
        image=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if image:
            caption,path=image.groups();img=Image(str(ROOT/path))
            ratio=width/img.imageWidth;img.drawWidth=width;img.drawHeight=img.imageHeight*ratio
            story.append(KeepTogether([Spacer(1,6),img,Spacer(1,5),Paragraph(inline(caption),styles['caption'])]));i+=1;continue
        if line.startswith('|'):
            cells=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                parts=[p.strip().replace('\\|','|') for p in re.split(r'(?<!\\)\|',lines[i].strip())[1:-1]]
                if not all(re.fullmatch(r':?-+:?',p) for p in parts):cells.append(parts)
                i+=1
            cols=len(cells[0]);assert all(len(r)==cols for r in cells),'Malformed manuscript table'
            if cols==2:widths=[125,width-125]
            elif cols==3:widths=[106,184,width-290]
            elif cols==8:widths=[91]+[49]*6+[119]
            elif cols==7:widths=[101]+[60]*5+[103]
            elif cols==6:widths=[121]+[(width-121)/5]*5
            elif cols==5 and cells[0][0]=='Model':widths=[107,49,49,149.5,149.5]
            elif cols==5:widths=[105,126,76,76,121]
            else:widths=[width/cols]*cols
            widths=[x*width/sum(widths) for x in widths]
            content=[[Paragraph(inline(c),styles['header' if ri==0 else 'cell']) for c in r] for ri,r in enumerate(cells)]
            tab=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT',spaceBefore=3,spaceAfter=12)
            tab.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.8,colors.HexColor('#243748')),
                                    ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#243748')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#243748')),
                                    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f7')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),
                                    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.append(tab);continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|```)',lines[i].strip()):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        kind='ref' if references else 'meta' if text.startswith('*Manuscript') else 'caption' if text.startswith('**Table ') else 'body'
        style=styles[kind]
        if kind=='caption':style=ParagraphStyle('table-caption',parent=style,keepWithNext=True)
        story.append(Paragraph(inline(text),style))

    destination=OUTPUT/'MEMLEARN_manuscript_v5.pdf'
    def page(canvas,doc):
        canvas.saveState();canvas.setFont(font,8);canvas.setFillColor(colors.HexColor('#58616c'))
        if doc.page>1:canvas.drawString(54,letter[1]-32,'Binding Makes Rules Addressable')
        canvas.drawRightString(letter[0]-54,30,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(destination),pagesize=letter,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=48,
                          title='Binding Makes Rules Addressable: Key-Mediated Retrieval of In-Context Rules',author='',
                          subject='MEMLEARN manuscript v5; existing experiments, corrected CPU analysis',pageCompression=1)
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(destination)
    return destination


if __name__=='__main__':
    make_figures()
    make_pdf()
