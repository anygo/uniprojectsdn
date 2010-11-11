//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 9
//
//  Write a program to read two 'std::string's and report the sizes of the 'std::string's
//  and whether the 'std::string's are equal. If not, report which of the two is the larger.
//
//=================================================================================================


#include <iostream>
#include <string>


int main(int argc, char* arv[]) {

	std::string s1,s2;
	std::cin >> s1 >> s2;

	std::cout << s1.size() << " " << s2.size() << std::endl;
	if(s1 != s2) {
		if(s1.size() > s2.size())
			std::cout << 1 << std::endl;
		else
			std::cout << 2 << std::endl; 	
	}
	else
			std::cout << "equal" << std::endl;


	
}

// KLAUS: ok, but very uninformative outputs ;-)
