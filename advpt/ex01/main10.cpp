//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 10
//
//  Write a program to strip the punctuation from a 'std::string'. The input to the program
//  should be a string of characters including punctuation; the output should be a 'std::string'
//  in which the punctuation is removed.
//
//=================================================================================================


#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>


struct IsPunct {
   inline bool operator()( const char c ) { return ispunct(c); }
};


int main(int argc, char* arg[]) {
		std::string line;

		std::cin >> line;
		line.erase(remove_if(line.begin(), line.end(), IsPunct() ), line.end());

		std::cout << line << std::endl;
}

// KLAUS: ok!
