//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 5 - Task 8
//
//  Write the function template 'findMostFrequentElement' that takes a pair of values that
//  represent iterators of unknown type. Find the value that occurs most frequently in the
//  sequence. Test your implementation with a 'vector<int>' and a 'list<string>'.
//
//=================================================================================================


#include <vector>
#include <list>
#include <map>
#include <iostream>


template<class Iter> typename Iter::value_type findMostFrequentElement(Iter first, Iter last) {

		std::map<typename Iter::value_type, int> elems;
		for ( ; first != last; ++first) {
				if (elems.find(*first) == elems.end()) 
						elems.insert(std::pair<typename Iter::value_type, int>(*first, 1));
				else
						elems[*first]++;
		}

		typename Iter::value_type max;
		int max_cnt = 0;
		typename std::map<typename Iter::value_type, int>::iterator it;
		for (it = elems.begin(); it != elems.end(); ++it) {
				if ((*it).second > max_cnt) {
						max_cnt = (*it).second;
						max = (*it).first;
				}
		}

		return max;
}

int main(int argc, char *argv[]) {
		std::vector<int> v;
		v.push_back(1);
		v.push_back(2);
		v.push_back(2);
		v.push_back(2);
		v.push_back(2);
		v.push_back(1);
		v.push_back(1);
		v.push_back(2);

		std::cout << findMostFrequentElement(v.begin(), v.end()) << std::endl;
}
