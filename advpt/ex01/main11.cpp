//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 11
//
//  Write a program to create a 'std::vector<int>' with 10 elements of your choice. Print the
//  initial elements. Using an iterator, assign each element a value that is twice its current
//  value. Afterwards, print the content of the vector again (using 'const_iterator's).
//
//=================================================================================================

#include <iostream>
#include <vector>
#include <algorithm>

// KLAUS: Note the additional header
#include <iterator>

using namespace std;

inline void myfunc (int &i)
{
	i*=2;
}

inline void myfunc2 (int i)
{
		std::cout << i << " ";
}

int main()
{
	std::vector<int> v;
	for (int i = 0; i < 10; i++)
			v.push_back(i);

	// KLAUS: As an alternative to the use of the for_each function, you can also "copy" the
	//        values from the first range to a second range specified by an ostream iterator.
	//        This makes the output even more concise.
	copy( v.begin(), v.end(), ostream_iterator<int>( cout, " " ) );
	cout << endl;

	for_each(v.begin(), v.end(), myfunc2);
	std::cout << std::endl;

	for_each(v.begin(), v.end(), myfunc);

	for_each(v.begin(), v.end(), myfunc2);
	std::cout << std::endl;
}

// KLAUS: ok!
