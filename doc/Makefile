DOC=main
PDF=$(DOC).pdf

TEX=pdflatex
TEXFLAGS=-halt-on-error

$(PDF):$(DOC).tex
	$(TEX) $(TEXFLAGS) $(DOC)

clean:
	rm -f $(DOC).aux $(DOC).log $(PDF) *.pdf

view:$(PDF)
	okular $(PDF)

.PHONY: clean view
