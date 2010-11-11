//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 1
//
//  Write a program that prompts the user for two numbers and writes each number in the range
//  specified by the two numbers to the standard output.
//
//=================================================================================================

#include <iostream>

int main(int argc, char *argv[])
{
		int a, b;
		std::cin >> a >> b;

		// KLAUS: Although syntactically correct, the postfix increment is sematically wrong in
		//        this case. Get used to using the prefix increment per default and only use the
		//        postfix increment in cases you require the "old" value. Admittedly, it hardly
		//        makes a difference here, but it might result in inefficient code for other
		//        data types.
		for (int i = std::min(a,b); i <= std::max(a,b); i++)
				std::cout << i << " ";
		std::cout << std::endl;

		return 0;
}

// KLAUS: ok!
