#include <iostream>

int calc(char *, char *) {return 1;}
int calc(char* const, char* const) {return 2;}

int main(int argc, char* argv[]) {

	char * x = "abc";
	char * const y = x;

	std::cout << calc(x,x);
	std::cout << calc(y, y);
}
