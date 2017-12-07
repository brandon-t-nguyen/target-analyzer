DOC=main
PDF=$(DOC).pdf

TEX=pdflatex
TEXFLAGS=-halt-on-error

BIB=bibtex

$(PDF):$(DOC).tex
	$(TEX) $(TEXFLAGS) $(DOC)
	$(BIB) $(DOC)
	$(TEX) $(TEXFLAGS) $(DOC)
	$(TEX) $(TEXFLAGS) $(DOC)

clean:
	rm -f $(DOC).aux $(DOC).log $(PDF) *.pdf *.bbl *.blg

view:$(PDF)
	okular $(PDF)

spell:
	aspell -t -c $(DOC).tex

.PHONY: clean view spell
