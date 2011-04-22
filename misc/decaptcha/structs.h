#ifndef STRUCTS_H
#define STRUCTS_H

#include "cv.h"
#include <vector>

struct CaptchaChar {
	cv::Mat1f char_img;
	cv::Mat1f feature_vector;
	char label;
};

struct Captcha {
	cv::Mat1f orig_img;
	cv::Mat1f processed_img;
	std::vector<CaptchaChar> extracted_chars;
};

#endif // STRUCTS_H