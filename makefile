SHELL := /bin/bash

.PHONY: clean test testprofile testcover docs

flist = 1 4 5 6 8 9 11 12 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S16 111
flistPath = $(patsubst %, output/figure%.svg, $(flist))

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: genFigures.py lineage/figures/figure%.py
	if test -r "$@"; then \
		touch $@; \
	else \
		poetry run ./genFigures.py $*; \
	fi

test:
	poetry run pytest -s -v -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports lineage

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	git clean -fdx output
