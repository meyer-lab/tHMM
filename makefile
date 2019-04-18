SHELL := /bin/bash
fdir = ./manuscript/figures
tdir = ./manuscript/templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown ./manuscript/text/*.md

.PHONY: clean test testprofile testcover docs

flist = 1 2 3 4 5 S1 S2 S3 S4 S5

$(fdir)/figure%.svg: genFigures.py 
	mkdir -p ./manuscript/figures
	./genFigures.py $*

$(fdir)/figure%pdf: $(fdir)/figure%svg
	rsvg-convert -f pdf $< -o $@

$(fdir)/figure%eps: $(fdir)/figure%svg
	rsvg-convert -f eps $< -o $@

manuscript/manuscript.pdf: manuscript/manuscript.tex $(patsubst %, $(fdir)/figure%.pdf, $(flist))
	(cd ./manuscript && latexmk -xelatex -f -quiet)
	rm -f ./manuscript/manuscript.b* ./manuscript/manuscript.aux ./manuscript/manuscript.fls

manuscript/manuscript.tex: manuscript/text/*.md
	pandoc -s $(pan_common) --template=$(tdir)/default.latex --pdf-engine=xelatex -o $@

manuscript/index.html: manuscript/text/*.md $(patsubst %, $(fdir)/figure%.svg, $(flist))
	pandoc -s $(pan_common) -t html5 --mathjax -c ./templates/kultiad.css --template=$(tdir)/html.template -o $@

test:
	pytest -s

testcover:
	pytest -s --junitxml=junit.xml --cov=lineage --cov-report xml:coverage.xml

testprofile:
	python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats profile | dot -Tsvg -o profile.svg

clean:
	rm -f coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg
	rm -rf prof manuscript/genFigures

docs:
	sphinx-apidoc -o doc/source lineage
	sphinx-build doc/source doc/build
