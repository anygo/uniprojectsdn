#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

std::string& to_uppercase(std::string& str) {

		std::transform(str.begin(), str.end(), str.begin(), toupper);
		return str;	
}

int main(int argc, char* argv[]) {
		
		std::vector<std::string> v;
		std::string tmp;

		while (std::cin >> tmp)
				v.push_back(tmp);

		std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(std::cout, " "));
		std::cout << std::endl;

		std::transform(v.begin(), v.end(), v.begin(), to_uppercase);

		std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(std::cout, " "));
		std::cout << std::endl;
		
		std::copy(v.rbegin(), v.rend(), std::ostream_iterator<std::string>(std::cout, " "));
		std::cout << std::endl;
}
