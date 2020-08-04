SHELL := /bin/bash

.PHONY: clean test testprofile testcover docs

flist = 1 2 4 5 6 7 S02 S03 S22 S23 S04 S05 S24 S25

all: spell.txt $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py lineage/figures/figure%.py
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml

test: venv
	. venv/bin/activate; pytest -s

spell.txt: manuscript/*.md
	pandoc --lua-filter common/templates/spell.lua manuscript/*.md | sort | uniq -ic > spell.txt

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg
