#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <algorithm>

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestRunner.h>

#include "tree.hpp"

using namespace std;

class interfaceTestCase : public CppUnit::TestCase {
public:
	// method to create a suite of tests
	static CppUnit::Test *suite() {
		CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("interfaceTestCase");

		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testCellInit", &interfaceTestCase::testCellInit));
		suiteOfTests->addTest(new CppUnit::TestCaller<interfaceTestCase>("testCellDoubleEndThrow", &interfaceTestCase::testCellDoubleEndThrow));

		return suiteOfTests;
	}

protected:
	void testCellInit() {
		cell cellTest(1.0, 1);

		CPPUNIT_ASSERT(cellTest.tstart == 1);
	}

	void testCellDoubleEndThrow() {
		cell cellTest(1.0, 1);

		cellTest.setDead(2.0);

		CPPUNIT_ASSERT_THROW(cellTest.setDivided(2.0, {{3, 4}}), std::runtime_error);
	}
};

// the main method
int main () {
	CppUnit::TextUi::TestRunner runner;

	ofstream outputFile("testResults.xml");
	CppUnit::XmlOutputter* outputter = new CppUnit::XmlOutputter(&runner.result(), outputFile);    
	runner.setOutputter(outputter);

	runner.addTest(interfaceTestCase::suite());
	
	runner.run();

	outputFile.close();

	return 0;
}