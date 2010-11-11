//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 3
//
//=================================================================================================

#include <iostream>

int main(int argc, char *argv[])
{
		int a;
		std::cin >> a;
		int fak = 1;

		// KLAUS: Although the program runs stable with negative input values, the only correct
		//        response to a negative input value is an error message.

		for (int i = 2; i <= a; i++)
				fak *= i;

		std::cout << fak << std::endl;

}

// KLAUS: ok!
