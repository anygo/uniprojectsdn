//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 3 - Task 5
//
//  Write a function that returns 0 when it is first called and then generates numbers
//  in sequence each time it is called again. Use this function in a suitable 'main'
//  function to demonstrate its correctness.
//
//=================================================================================================

#include <iostream>

static int bla = 0;

int f() {
		bla++;
		return bla-1;
}

int main(int argc, char *argv[]) {
		for (int i = 0; i < 42; ++i) {
			std::cout << "Aufruf " << i << ": " << f() << std::endl;
		}

		return 0;
}
