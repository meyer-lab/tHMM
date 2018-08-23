

cppcheck: lineage/cppcheck.cpp lineage/tree.hpp
	g++ -L/usr/lib/x86_64-linux-gnu/ -L/usr/lib/gcc/x86_64-linux-gnu/7/ -I/usr/include/cppunit/ -g --std=c++17 -coverage -Wall -Wextra lineage/cppcheck.cpp -lstdc++ -lcppunit -o $@

test: cppcheck
	./cppcheck
	lcov -c -d ./ -o coverage.info --no-external -q
	genhtml coverage.info --output-directory coverage.html

clean:
	rm -f testResults.xml cppcheck cppcheck.gcda cppcheck.gcno coverage.info
	rm -rf cppcheck.dSYM coverage.html