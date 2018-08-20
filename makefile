

cppcheck: lineage/cppcheck.cpp lineage/tree.hpp
	clang++ -g --std=c++17 -mavx -march=native -Wall -Wextra -lcppunit lineage/cppcheck.cpp -o $@

clean:
	rm -f testResults.xml cppcheck
