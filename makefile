SHELL := /bin/bash
fdir = ./manuscript/figures
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=./common/templates/figure-filter.py -f markdown ./manuscript/*.md

.PHONY: clean test testprofile testcover docs

flist = 1 2 3 4 5 6 7

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv --system-site-packages venv
	. venv/bin/activate && pip install -Ur requirements.txt
	touch venv/bin/activate

$(fdir)/figure%.svg: venv genFigures.py 
	mkdir -p ./manuscript/figures
	. venv/bin/activate && ./genFigures.py $*

$(fdir)/figure%pdf: $(fdir)/figure%svg
	rsvg-convert -f pdf $< -o $@

$(fdir)/figure%eps: $(fdir)/figure%svg
	rsvg-convert -f eps $< -o $@

manuscript/manuscript.pdf: manuscript/manuscript.tex $(patsubst %, $(fdir)/figure%.pdf, $(flist))
	(cd ./manuscript && latexmk -xelatex -f -quiet)
	rm -f ./manuscript/manuscript.b* ./manuscript/manuscript.aux ./manuscript/manuscript.fls

manuscript/manuscript.tex: manuscript/*.md
	pandoc -s $(pan_common) --template=./common/templates/default.latex --pdf-engine=xelatex -o $@

Guide_to_tHMM.pdf: venv Guide_to_tHMM.ipynb
	. venv/bin/activate && jupyter nbconvert --to pdf --execute Guide_to_tHMM.ipynb

test: venv
	. venv/bin/activate; pytest -s

testcover: venv
	. venv/bin/activate; pytest -s --junitxml=junit.xml --cov=lineage --cov-report xml:coverage.xml

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats --node-thres=2.0 profile | dot -Tsvg -o profile.svg

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc lineage > pylint.log || echo "pylint exited with $?")

clean:
	rm -f coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg
	rm -rf prof manuscript/figures

docs: venv
	. venv/bin/activate && sphinx-apidoc -o doc/source lineage
	. venv/bin/activate && sphinx-build doc/source doc/build
