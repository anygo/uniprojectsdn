//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 4 - Task 7
//
//  Using the following definition of 'ia', copy 'ia' into a 'vector' and into 'list'. Use the
//  single iterator form of 'erase' to remove the elements with odd values from your 'list' and
//  the even values from your 'vector'.
//
//  int ia[] = { 0, 1, 1, 2, 3, 5, 8, 13, 21, 55, 89 };
//
//=================================================================================================

#include <iostream>
#include <vector>
#include <list>

int main(int argc, char* argv[]) {
		
		int ia[] = {0,1,1,2,3,5,8,13,21,34,55,89};

		std::vector<int> v(ia, ia+12);
		std::list<int> l(ia, ia+12);

		for (std::vector<int>::iterator it = v.begin(); it != v.end();) {
				std::vector<int>::iterator here = it++;
				if (*here%2==0)
						v.erase(here);
		}

		for (std::list<int>::iterator it = l.begin(); it != l.end();) {
				std::list<int>::iterator here = it++;
				if (*here%2==1)
						l.erase(here);
		}


		for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
				std::cout << *it << " ";
		}
		std::cout << std::endl;

		for (std::list<int>::iterator it = l.begin(); it != l.end(); ++it) {
				std::cout << *it << " ";
		}
		std::cout << std::endl;
}
