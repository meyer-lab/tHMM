

cppcheck: lineage/cppcheck.cpp lineage/tree.hpp
	g++ -g --std=c++17 -coverage -mavx -march=native -Wall -Wextra -lcppunit lineage/cppcheck.cpp -o $@

test: cppcheck
	./cppcheck
	lcov -c -d ./ -o coverage.info --no-external -q
	genhtml coverage.info --output-directory coverage.html

clean:
	rm -f testResults.xml cppcheck cppcheck.gcda cppcheck.gcno coverage.info
	rm -rf cppcheck.dSYM coverage.html