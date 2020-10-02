SHELL := /bin/bash

.PHONY: clean test testprofile testcover docs

flist = 1 4 5 6 7 8 9 11 12 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10

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
	mkdir -p output/output
	cp output/*.svg output/output/
	cp -n lineage/figures/cartoons/figure*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml

output/manuscript.docx: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir -p output/output
	cp output/*.svg output/output/
	cp -n lineage/figures/cartoons/figure*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml

test: venv
	. venv/bin/activate; pytest -s -v -x

spell.txt: manuscript/*.md
	pandoc --lua-filter common/templates/spell.lua manuscript/*.md | sort | uniq -ic > spell.txt

testprofile: venv
	python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg
