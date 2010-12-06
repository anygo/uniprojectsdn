#include <iostream>


int main(int argc, char* argv[]) {
	int i = 1;
	int j = 1;

	int a = i++;
	int b = ++j;

	std::cout<< a << " " << b;

}
