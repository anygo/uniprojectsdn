#include <iostream>

int main(int argc, char *argv[]) {
	
		unsigned int from;
		unsigned int to;
		std::cin >> from;
		std::cin >> to;

		int sum = 0;
		for (int i = from; i < to; ++i) {
				sum = sum + i;
		}

		std::cout << sum << std::endl;
}
