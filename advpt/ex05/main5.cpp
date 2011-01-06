//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 5 - Task 5
//
//  Implement a generic 'bubblesort' function that sorts a range of elements specified via a
//  pair of iterators. This implementation should pose as few requirements as possible on the
//  element type of the range and on the iterators as possible! Test your implementation both
//  with a 'vector<int>' and a 'list<int>'.
//
//=================================================================================================

#include <iostream>
#include <vector>
#include <list>
#include <algorithm>

template<class Iterator> void bubblesort(Iterator start, Iterator end) {

	Iterator curr = start;

	bool count_iteration = true;

	int n = 2;

	for (bool swapped = true; swapped && n > 1; ) {

		if (count_iteration)
			n = 0;

		swapped = false;
		for(; curr != end; ) {

			if (count_iteration) 
				n++;
				
			Iterator prev = curr;
			++curr;
			
			if(curr == end) break;

			if( *prev > *curr ) {
				iter_swap(prev, curr);
				swapped = true;		
			}
		}

		curr = start;
		--n;
		count_iteration = false;
	}
}


int main(int argc, char* argv[]) {
	
	std::vector<int> v;
	v.push_back(4);
	v.push_back(5);
	v.push_back(0);
	v.push_back(2);
	v.push_back(1);
	v.push_back(3);

	for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
		std::cout << *it << " ";
	}
	std::cout << std::endl;

	bubblesort(v.begin(), v.end());

	for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
		std::cout << *it << " ";
	}
	std::cout << std::endl;



	// jetzt fuer listen
	std::list<int> l;
	l.push_back(4);
	l.push_back(5);
	l.push_back(0);
	l.push_back(2);
	l.push_back(1);
	l.push_back(3);

	for (std::list<int>::iterator it = l.begin(); it != l.end(); ++it) {
		std::cout << *it << " ";
	}
	std::cout << std::endl;

	bubblesort(l.begin(), l.end());

	for (std::list<int>::iterator it = l.begin(); it != l.end(); ++it) {
		std::cout << *it << " ";
	}
	std::cout << std::endl;

}
