//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 5 - Task 4
//
//  Write a template that acts like the library 'find' algorithm. Your template should take
//  a single type parameter that will name the type for a pair of iterators that should be
//  parameters to the function. Use your function to find a given value in a 'vector<int>'
//  and in a 'list<string>'.
//
//=================================================================================================

#include <string>
#include <list>
#include <vector>
#include <iostream>
#include <iterator>

template<class Iterator> Iterator find(Iterator start, Iterator end, iterator_traits<Iterator>::value_type val) {
	while(start != end) {
		if (*start == val)
				break;
		++start;
	}
	return start;
}


int main(int argc, char* argv[]) {

	std::vector<int> intvec(100,10);
	std::list<std::string> list;

	intvec[23] = 42;
	list.push_back("Spaten");
	list.push_back("Garten");
	list.push_back("Mausefalle");
	list.push_back("Hundefaust");
	list.push_back("Garten");

	std::cout << *(find(list.begin(), list.end(), "Garten")) << std::endl;
	std::cout << *(++find(list.begin(), list.end(), "Garten")) << std::endl;
	std::cout << *(find(list.begin(), list.end(), "Fausthund")) << std::endl;


	std::cout << *(find(intvec.begin(), intvec.end()  , 42)) << std::endl;
	std::cout << *(find(intvec.begin(), intvec.end()  , 23)) << std::endl;
	std::cout << *(find(intvec.begin(), intvec.end()-1, 23)) << std::endl;

}



