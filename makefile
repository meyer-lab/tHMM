

test:
	nosetests3 --with-xunit -s --with-timer --timer-top-n 20

testcover:
	nosetests3 --with-xunit --with-xcoverage --cover-package=lineage -s --with-timer --timer-top-n 20

clean:
	rm -f nosetests.xml coverage.xml .coverage

docs: 
	doxygen doxygen.cfg