//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 4 - Task 8
//
//  Create an arbitrarily filled 'set' of integers. Print the elements of the 'set'.
//  Afterwards remove all even values from the 'set' and reprint it.
//
//=================================================================================================

#include <set>
#include <iostream>
#include <algorithm>
#include <cstdlib>

bool isEven(int i) { return (i%2) == 0; }

int main(int argc, char* argv[]) {
		std::set<int> myset;

		std::set<int>::iterator it;

		for (int i = 0; i < 12; ++i) {
				int num = std::rand();
				myset.insert(num);	
				std::cout << num << " inserted" << std::endl;
		}		
		std::cout << std::endl;

		for (it = myset.begin(); it != myset.end();) {
				std::set<int>::iterator here = it++;
				if (isEven(*here))
						myset.erase(here);
		}


		for (it = myset.begin(); it != myset.end(); ++it) {
				std::cout << *it << std::endl;
		}
}
