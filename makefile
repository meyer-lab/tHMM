SHELL := /bin/bash

.PHONY: clean test testprofile testcover docs

all: output/manuscript.html pylint.log spell.txt

flist = 0 1 2 3 4 5 6 7 8 9 12 13 14 15 23

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py 
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=content --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose --data-dir=common/templates/pandoc \
		--defaults=common.yaml --defaults=html.yaml output/manuscript.md

Guide_to_tHMM.pdf: venv Guide_to_tHMM.ipynb
	. venv/bin/activate && jupyter nbconvert --to pdf --execute Guide_to_tHMM.ipynb

test: venv
	. venv/bin/activate; pytest -s

spell.txt: manuscript/*.md
	pandoc --lua-filter common/templates/spell.lua manuscript/*.md | sort | uniq -ic > spell.txt

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc lineage > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg pylint.log
