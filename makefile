

cppcheck: lineage/cppcheck.cpp lineage/tree.hpp
	clang++ -g --std=c++17 -mavx -march=native -Wall -Wextra -lcppunit lineage/cppcheck.cpp -o $@


test: cppcheck
	./cppcheck
	lcov --capture --directory lineage/ --output-file coverage.info --no-external -q
	genhtml coverage.info --output-directory coverage.html

clean:
	rm -f testResults.xml cppcheck
	rm -rf cppcheck.dSYM
