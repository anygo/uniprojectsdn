//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 4 - Task 2
//
//  Write a program to read in a file containing several words, sort them alphabetically, and
//  write them to a second file. Note that both filenames should be specified on the command
//  line: './main2 file1.in file2.out'.
//
//=================================================================================================

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

int main(int argc, char* argv[]) {
	
		if (argc != 3) {
				std::cout << "Programm richtig aufrufen!" << std::endl;
				return -1;
		}

		std::ifstream in(argv[1]);
		std::ofstream out(argv[2]);
		if (!in.is_open() || !out.is_open()) {
				std::cout << "Datei kann nicht geoeffnet werden" << std::endl;
				return -1;
		}



		std::vector<std::string> words;
		while (!in.eof()) {
			std::string word;
			in >> word;
			if (word.empty())
					continue;
			words.push_back(word);
		}

		in.close();

		sort(words.begin(), words.end());

		for (unsigned int i = 0; i < words.size(); ++i) {
				out << words[i] << "\n";
		}


		out.close();
}

