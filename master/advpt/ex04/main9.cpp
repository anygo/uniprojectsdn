//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 4 - Task 9
//
//  Create a 'set' of unsigned integers sorted according to the following rule: the elements
//  are sorted according to the first digit of the value. In case two values have the same
//  first digit, they are sorted according to the second digit, etc. Values with a smaller
//  number of digits are sorted before values with a higher number of digits. Fill the 'set'
//  with appropriate values and print all elements. Afterwards, erase all values, whose first
//  digit is 1 and reprint the set.
//
//=================================================================================================


#include <iostream>
#include <sstream>
#include <set>
#include <cstdlib>

struct comp {
		bool operator() (const unsigned int& lhs, const unsigned int& rhs) {
				std::stringstream ss;
				ss << lhs;
				std::stringstream ss2;
				ss2 << rhs;

				
				std::string l = ss.str();
				std::string r = ss2.str();

				
				//we could have done some awesome stuff:
				//Math.log(Math.abs(lhs)) + 1
				//... whatever
				if (r.size() == l.size()) {
						return l < r;
				} else {
						return l.size() < r.size();
				}
		}
};

int main(int argc, char* argv[]) {
		std::set<unsigned int, comp> s;

		for (int i = 0; i < 15; ++i) {
				s.insert(std::rand()%10000);
		}	

		for (std::set<unsigned int, comp>::iterator it = s.begin(); it != s.end(); ++it) {
				std::cout << *it << " ";
		}
		std::cout << std::endl;

		for (std::set<unsigned int, comp>::iterator it = s.begin(); it != s.end();) {
				std::set<unsigned int, comp>::iterator here = it++;
				
				std::stringstream ss;
				ss << *here;
				if (ss.str().c_str()[0] == '1')
						s.erase(here);
		}


		for (std::set<unsigned int, comp>::iterator it = s.begin(); it != s.end(); ++it) {
				std::cout << *it << " ";
		}
		std::cout << std::endl;
}
