SHELL := /bin/bash

.PHONY: clean test testprofile testcover docs

flist = 1 4 5 6 8 9 11 12 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S16
flistPath = $(patsubst %, output/figure%.svg, $(flist))

all: spell.txt $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py lineage/figures/figure%.py
	if test -r "$@"; then \
		touch $@; \
	else \
		. venv/bin/activate && JAX_PLATFORM_NAME=cpu ./genFigures.py $*; \
	fi

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(flistPath)
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistPath)
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml output/manuscript.md

test: venv
	. venv/bin/activate; pytest -s -v -x

mypy: venv
	. venv/bin/activate; mypy --install-types -r requirements.txt
	. venv/bin/activate; mypy --non-interactive --ignore-missing-imports lineage

spell.txt: manuscript/*.md
	pandoc --lua-filter common/templates/spell.lua manuscript/*.md | sort | uniq -ic > spell.txt

testprofile: venv
	python3 -m cProfile -o profile /usr/local/bin/pytest
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	git clean -fdx output
