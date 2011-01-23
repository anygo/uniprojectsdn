//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 4 - Task 3
//
//  Write a function to produce the Fibonacci sequence. Use this function to fill a
//  'vector<string>' with the first ten Fibonacci numbers. Sort the vector lexicographically
//  and print the resulting sequence to the screen.
//
//=================================================================================================


#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

int fib(int num) {
		if (num == 0 || num == 1)
				return 1;
		else 
				return fib(num-1) + fib(num-2);
}

int main(int argc, char* argv[]) {

		std::vector<std::string> fibs;

		for (int i = 0; i < 10; ++i) {
				int cur = fib(i);
				std::stringstream ss;
				ss << cur;
				fibs.push_back(ss.str());
		}

		sort(fibs.begin(), fibs.end());

		for (int i = 0; i < 10; ++i) {
				std::cout << fibs[i] << std::endl;
		}
}
