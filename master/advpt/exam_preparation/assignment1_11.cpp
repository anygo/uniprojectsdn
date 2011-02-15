#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <iterator>

int times_two(int x) { return x << 1; }

int main(int argc, char* argv[]) {
	
		typedef std::vector<int> Ints;
		
		Ints v;
		for (int i = 0; i < 10; i++) {
				v.push_back(rand() % 10);
		}

		for (Ints::const_iterator it = v.begin(); it != v.end(); ++it) {
				std::cout << *it << " ";
		} 
		std::cout << std::endl;

		transform(v.begin(), v.end(), v.begin(), times_two);
		
		for (Ints::const_iterator it = v.begin(); it != v.end(); ++it) {
				std::cout << *it << " ";
		} 
		std::cout << std::endl;
		
		transform(v.begin(), v.end(), v.begin(), std::bind1st(std::multiplies<int>(), 2));
		
		for (Ints::const_iterator it = v.begin(); it != v.end(); ++it) {
				std::cout << *it << " ";
		} 
		std::cout << std::endl;

		std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
}
