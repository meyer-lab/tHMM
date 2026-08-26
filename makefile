SHELL := /bin/bash

.PHONY: clean test check lint format testprofile docs ty
flist = $(wildcard lineage/figures/figure*.py)

all: $(patsubst lineage/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: lineage/figures/figure%.py
	if test -r "$@"; then \
		touch $@; \
	else \
		uv run fbuild $*; \
	fi

test:
	uv run pytest -v -s -x

ty:
	uv run ty check

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint ty

testprofile:
	uv run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	git clean -fdx output

