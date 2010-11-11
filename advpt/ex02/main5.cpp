//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 5
//
//  Write a program to create a dynamically allocated array for 10 'int's. Assign each element
//  an 'int' of your choice. Copy this array into a second dynamically allocated array. Afterwards,
//  compare the two arrays element-wise. Print both arrays.
//
//=================================================================================================


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[])
{
		int *arr = new int[10];
		for (int i = 0; i < 10; i++)
				arr[i] = i*i*3;

		int *arr2 = new int[10];
		copy(arr, arr+10, arr2);

		for (int i = 0; i < 10; i++)
				if (arr[i] != arr2[i]) 
						std::cout << i << "te Stelle verschieden" << std::endl;


		std::cout << "Original: ";
		for (int i = 0; i < 10; i++)
				std::cout << arr[i]  << " ";
		std::cout << std::endl;

		std::cout << "Arr2: ";
		for (int i = 0; i < 10; i++)
				std::cout << arr2[i] << " ";
		std::cout << std::endl;




		delete [] arr2;
		delete [] arr;

		return 0;
}
