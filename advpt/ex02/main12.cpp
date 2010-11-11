//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 2 - Task 12
//
//  The 'std::vector' function 'at' returns a reference to the accessed element. In case the
//  element does not exist, the function throws a 'std::out_of_range' exception that is derived
//  from 'std::logic_error'. Write code to provoke this error, catch the exception as
//  'std::logic_error', and print the error message on the screen.
//
//=================================================================================================


#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

int main(int argc, char* argv[])
{
		vector<int> v(10,42);
		
		for (int i = 0; i < 15; i++)
		{
			try
			{
				std::cout << v.at(i) << std::endl;
			}
			catch (std::logic_error &e)
			{
				std::cout << e.what() << std::endl;
			}
		}
		return 0;
}
