#include <iostream>
#include <list>
#include <vector>
#include <map>

#include "cv.h"
#include "highgui.h"
#include "ml.h"

#include "decaptcha_tool.h"


int main(int args, char *argv[]) {

	std::cout << "Initializing DecaptchaTool..." << std::endl;
	DecaptchaTool tool;

	std::cout << "How many images to use? (current max: 36) ";
	unsigned int num;
	std::cin >> num;

	if (num > 36) num = 36;

	// read some captcha images
	for (int im_num = 1; im_num <= num; ++im_num) {

		std::stringstream ss;
		ss << "img/" << im_num << ".jpg";
		std::string im_name = ss.str();

		tool.add_image(im_name);
		std::cout << " -> added " << im_name << std::endl;
	}
	std::cout << std::endl;


	std::cout << "(1) train or (2) test a classifier? (1/2) ";
	char c;
	std::string str;
	std::cin >> c;
	switch (c) {
		case '1': 
			std::cout << "filename for new classifier (e.g. classifier.xml): ";
			std::cin >> str;
			tool.train(str);
			break;
		case '2': 
			std::cout << "filename of existing classifier (e.g. classifier.xml / all.xml): ";
			std::cin >> str;
			tool.test(str);
			break;
		default:
			std::cout << " !!!!! error: type '1' or '2'" << std::endl;
	}
}