#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main(int argc, char* argv[]) {
	
		
		std::cout << "std::vector<int> v(23)" << std::endl;
		std::vector<int> v(23);
		std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << " <- " << v.size() << "/" << v.capacity() << std::endl;

		std::cout << "v.push_back(1)" << std::endl;
		v.push_back(2);
		std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << " <- " << v.size() << "/" << v.capacity() << std::endl;

		std::cout << "std::vector<int>(v).swap(v)" << std::endl;
		std::vector<int>(v).swap(v);
		std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << " <- " << v.size() << "/" << v.capacity() << std::endl;
		
		std::cout << "std::vector<int>().swap(v)" << std::endl;
		std::vector<int>().swap(v);
		std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << " <- " << v.size() << "/" << v.capacity() << std::endl;
}
