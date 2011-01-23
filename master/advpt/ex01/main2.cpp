//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 2
//
//  Write a program to sum the numbers in a user-specified range, omitting the 'if' test that
//  sets the upper and lower bounds. Predict what happens if the input is the numbers 7 and 3,
//  in that order. Now run the program giving it the numbers 7 and 3, and see if the results
//  match your expectation. If not, restudy the behavior of 'for' and 'while' loops until you
//  understand what happened.
//
//=================================================================================================

#include <iostream>

int main(int argc, char *argv[])
{
		int a,b;
		std::cin >> a >> b;
		int sum = 0;
		for (int i = a; i < b; i++)
				sum += i;

		std::cout << sum << std::endl;

}

// KLAUS: ok!
