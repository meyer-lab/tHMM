

test:
	nosetests3 --with-xunit -s --with-timer --timer-top-n 20

clean:
	echo "Clean stub."

docs: 
	doxygen doxygen.cfg