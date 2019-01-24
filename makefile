
test:
	nosetests3 --with-xunit -s --with-timer --timer-top-n 20

testcover:
	nosetests3 --with-xunit --with-cprofile --with-xcoverage --cover-package=lineage -s --with-timer --timer-top-n 20
	gprof2dot -f pstats stats.dat | dot -Tsvg -o cprofile.svg

clean:
	rm -f nosetests.xml coverage.xml .coverage stats.dat cprofile.svg

docs:
	doxygen doxygen.cfg