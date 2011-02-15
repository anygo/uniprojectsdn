#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char* argv[]) {
		
		std::string input;
		std::cin >> input;
		
		input.erase(remove_if(input.begin(), input.end(), ispunct), input.end());

		std::cout << input << std::endl;
}
