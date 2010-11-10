SHELL=/bin/bash
TARGET = vortrag

EPS= $(wildcard eps/*.eps)
JPG= $(wildcard jpg/*.jpg)
FIG= $(wildcard fig/*.fig)
TEXFIG= $(wildcard texfig/*.fig)

EPS_C= $(EPS:eps/%.eps=eps_c/%.pdf)
JPG_C= $(JPG:jpg/%.jpg=jpg_c/%.eps)
FIG_PS = $(FIG:fig/%.fig=fig_c/%.eps)
FIG_PDF= $(FIG:fig/%.fig=fig_c/%.pdf)
TEXFIG_TEX= $(TEXFIG:texfig/%.fig=texfig_c/%.pstex_t)
TEXFIG_PS = $(TEXFIG:texfig/%.fig=texfig_c/%.pstex)
TEXFIG_PDF= $(TEXFIG:texfig/%.fig=texfig_c/%.pdf)

.PHONY: all clean cleanall pdf pdf! ps ps!

# Bilder-Konvertierung: eps -> pdf
eps_c/%.pdf: eps/%.eps
	@echo Konvertiere $< nach $@
	GS_OPTIONS=-dAutoRotatePages=/None epstopdf $< -o $@

# Bilder-Konvertierung: jpg -> eps
jpg_c/%.eps: jpg/%.jpg
	@echo Konvertiere $< nach $@
	jpeg2ps $< > $@

# Bilder-Konvertierung: fig -> eps
fig_c/%.eps: fig/%.fig
	@echo Konvertiere $< nach $@eps
	fig2dev -L eps $< > $@

# Bilder-Konvertierung: fig -> pdf
fig_c/%.pdf: fig_c/%.eps
	@echo Konvertiere $< nach $@
	GS_OPTIONS=-dAutoRotatePages=/None epstopdf $< -o $@

# Biler-Konvertierung: Combined LaTeX / PostScript (PDF)
texfig_c/%.pstex_t: texfig/%.fig
	@echo Konvertiere $< nach $@
	fig2dev -L pstex_t $< | ./scripts/modify_pstex_t.pl $< > $@
texfig_c/%.pstex: texfig/%.fig
	@echo Konvertiere $< nach $@
	fig2dev -L pstex $< > $@
texfig_c/%.pdf: texfig_c/%.pstex
	@echo Konvertiere $< nach $@
	GS_OPTIONS=-dAutoRotatePages=/None epstopdf --nocompress $< -o $@

eps_c/%.pdf: eps/%.eps
	@echo Konvertiere $< nach $@
	epstopdf $< -o $@

# Bilder-Konvertierung: jpg-> eps
jpg_c/%.eps: jpg/%.jpg
	@echo Konvertiere $< nach $@
	jpeg2ps $< > $@


all: pdf

pdf: $(TARGET).pdf

pdf!: cleanall pdf

$(TARGET).pdf: $(EPS_C) $(JPG) $(FIG_PDF) $(TEXFIG_TEX) $(TEXFIG_PDF) $(TARGET).tex
	pdflatex  $(TARGET).tex
	pdflatex  $(TARGET).tex

ps: $(TARGET).ps

ps!: cleanall ps

$(TARGET).ps: $(TARGET).dvi
	dvips -P pdf $(TARGET).dvi
	pstops '0@0.92(0.8cm,1.188cm)' vortrag.ps > tmp.ps
	mv tmp.ps $(TARGET).ps

$(TARGET).dvi: $(TARGET).tex $(EPS) $(JPG_C) $(FIG_PS) $(TEXFIG_TEX) $(TEXFIG_PS)
	latex  $(TARGET).tex
	latex  $(TARGET).tex

clean:
	rm -f $(TARGET).dvi $(TARGET).log $(TARGET).aux $(TARGET).out $(TARGET).toc $(TARGET).snm $(TARGET).nav *~ *.vim

cleanall: clean
	rm -f $(TARGET).pdf $(TARGET).ps
