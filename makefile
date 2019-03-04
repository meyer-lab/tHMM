.PHONY: clean test testprofile testcover docs

test:
	pytest

testcover:
	pytest --junitxml=junit.xml --cov=lineage --cov-report xml:coverage.xml

testprofile:
	pytest --profile-svg

clean:
	rm -f coverage.xml .coverage .coverage* junit.xml coverage.xml
	rm -rf prof

docs:
	sphinx-apidoc -o doc/source lineage
	sphinx-build doc/source doc/build
