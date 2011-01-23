//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 12
//
//  Write a program, that reads in several words at once and stores them in a 'std::vector' of
//  'std::string's. Process all words by converting them to upper case strings and afterwards
//  print them in reverse order on the screen. Use both the subscript operator and iterators to
//  traverse the vector!
//
//=================================================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype>

inline void write_something (std::string &str)
{
		std::cout << str << " ";
}

int main()
{
		std::vector<std::string> vec;
		while (true)
		{
			std::string str;
			std::getline(std::cin, str);
			if (str.empty() || str == "\n")
					break;

			std::transform(str.begin(), str.end(), str.begin(), toupper);
			vec.push_back(str);

			std::cout << str << std::endl;
		}

		// KLAUS: Also here you could use the ostream_iterator-based output:
		// copy( vec.rbegin(), vec.rend(), ostream_iterator<string>( cout, " " ) );
		for_each(vec.rbegin(), vec.rend(), write_something);
}

// KLAUS: very nice solution!
