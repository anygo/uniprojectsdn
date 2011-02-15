#include <iostream>
#include <algorithm>
#include <iterator>

int cheating(int unused) { return rand() % 10; }

int main(int argc, char* argv[]) {

		int *ints = new int[10];
		
		std::transform(ints, ints+10, ints, cheating);

		std::copy(ints, ints+10, std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;

		
		int *ints_copy = new int[10];
		std::copy(ints, ints+10, ints_copy);
		
		std::copy(ints_copy, ints_copy+10, std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;

		delete [] ints_copy;
		delete [] ints;
}
