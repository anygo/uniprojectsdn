#ifndef CAPTCHA_PROCESSOR_H
#define CAPTCHA_PROCESSOR_H

#include <vector>
#include <string>
#include <list>
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include "structs.h"


// increase for "nice animations"
#define TIME_TO_WAIT 1



class CaptchaProcessor {
public:
	CaptchaProcessor(Captcha &captcha, bool verbose = true) : captcha_(captcha), verbose_(verbose) {}
	~CaptchaProcessor() {}

	void process();

protected:
	static void show_process_image(const cv::Mat1f &img);

	Captcha &captcha_;
	bool verbose_;
};

#endif // CAPTCHA_PROCESSOR_H