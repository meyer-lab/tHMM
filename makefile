
test:
	nosetests3 --with-xunit -s --with-timer

testcover:
	nosetests3 --processes=4 --process-timeout=60 --with-xunit --with-xcoverage --cover-package=lineage

testprofile:
	nosetests3 --with-cprofile
	gprof2dot -f pstats stats.dat | dot -Tsvg -o cprofile.svg

clean:
	rm -f nosetests.xml coverage.xml .coverage stats.dat cprofile.svg

docs:
	doxygen doxygen.cfg
