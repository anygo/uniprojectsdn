//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 6
//
//  Rewrite the previous program (main5.cpp), but this time use 'vector's instead of plain arrays.
//
//=================================================================================================


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[])
{
		vector<int> v0;
		for (int i = 0; i < 10; i++)
				v0.push_back(i*i*i*i);

		vector<int> v1(v0);


		for (int i = 0; i < 10; i++)
				if (v0[i] != v1[i]) 
						std::cout << i << "te Stelle verschieden" << std::endl;


		std::cout << "Original: ";
		for (int i = 0; i < 10; i++)
				std::cout << v0[i]  << " ";
		std::cout << std::endl;

		std::cout << "Kopie:    ";
		for (int i = 0; i < 10; i++)
				std::cout << v1[i] << " ";
		std::cout << std::endl;


		return 0;
}
