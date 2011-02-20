#include <iostream>
#include <vector>
#include <list>
#include <iterator>

int main(int argc, char* argv[]) {

		std::vector<int> v;

		for (int i = 0; i < 10; i++) v.push_back(i);

		for (std::vector<int>::iterator it = v.begin(); it != v.end(); /*do nothing*/) {
				
				// scheinbar wird von rechts nach links ausgewertet?!?!
				std::cout << *it << " " << *it++ << " " << *it << std::endl;
		}
}
