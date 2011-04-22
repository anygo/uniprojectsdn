#include "captcha_processor.h"


void CaptchaProcessor::process() {
	
	if (verbose_) std::cout << "thresholding..." << std::endl;
	cv::threshold(captcha_.processed_img, captcha_.processed_img, 0.5, 1, CV_THRESH_BINARY_INV);
	show_process_image(captcha_.processed_img);

	if (verbose_) std::cout << "cropping..." << std::endl;
	captcha_.processed_img = captcha_.processed_img(cv::Range(1, captcha_.processed_img.rows - 1), cv::Range(1, captcha_.processed_img.cols - 1));
	show_process_image(captcha_.processed_img);

	if (verbose_) std::cout << "detect & remove vertical lines..." << std::endl;
	std::list<int> vertical_lines;
	for (int i = 0; i < captcha_.processed_img.cols; ++i) {
		if (captcha_.processed_img(0, i) == 1) {
			captcha_.processed_img.col(i) = 0;
			vertical_lines.push_back(i);
			show_process_image(captcha_.processed_img);
		}
	}
	show_process_image(captcha_.processed_img);

	if (verbose_) std::cout << "detect & remove horizontal lines..." << std::endl;
	std::list<int> horizontal_lines;
	for (int i = 0; i < captcha_.processed_img.rows; ++i) {
		if (captcha_.processed_img(i, 0) == 1) {
			captcha_.processed_img.row(i) = 0;
			horizontal_lines.push_back(i);
			show_process_image(captcha_.processed_img);
		}
	}
	show_process_image(captcha_.processed_img);

	if (verbose_) std::cout << "refining vertically..." << std::endl;
	for (std::list<int>::const_iterator it = vertical_lines.begin(); it != vertical_lines.end(); ++it) {
		cv::Mat1f roi = captcha_.processed_img.colRange((*it)-1, (*it)+1);
		cv::dilate(roi, roi, cv::Mat1f::ones(1, 2));
		cv::erode(roi, roi, cv::Mat1f::ones(1, 2));
		show_process_image(captcha_.processed_img);
	}

	if (verbose_) std::cout << "refining horizontally..." << std::endl;
	for (std::list<int>::const_iterator it = horizontal_lines.begin(); it != horizontal_lines.end(); ++it) {
		captcha_.processed_img = captcha_.processed_img.t();
		cv::Mat1f roi = captcha_.processed_img.colRange((*it)-1, (*it)+1);
		cv::dilate(roi, roi, cv::Mat1f::ones(1, 2));
		cv::erode(roi, roi, cv::Mat1f::ones(1, 2));
		captcha_.processed_img = captcha_.processed_img.t();
		show_process_image(captcha_.processed_img);
	}
	
	if (verbose_) std::cout << "cropping..." << std::endl;
	captcha_.processed_img = captcha_.processed_img(cv::Range(1, captcha_.processed_img.rows - 1), cv::Range(1, captcha_.processed_img.cols - 1));
	show_process_image(captcha_.processed_img);

	std::vector<cv::Mat1f> chars;
	if (verbose_) std::cout << "extracting single chars..." << std::endl;
	for (int i = 0; i < captcha_.processed_img.cols; ++i) {
		while (i < captcha_.processed_img.cols && cv::countNonZero(captcha_.processed_img.col(i)) == 0) ++i; // skip empty columns
		if (i >= captcha_.processed_img.cols) break;
		int start = i;
		while (i < captcha_.processed_img.cols && cv::countNonZero(captcha_.processed_img.col(i)) > 0) ++i;
		int end = i;
		if (i >= captcha_.processed_img.cols) break;
		chars.push_back(captcha_.processed_img.colRange(start, end));
	}
	if (verbose_) std::cout << chars.size() << " chars found - " << ((chars.size() == 6) ? "OK" : "ERROR") << std::endl;
	show_process_image(captcha_.processed_img);

	if (verbose_) std::cout << "compute feature vectors for single chars..." << std::endl;
	for (std::vector<cv::Mat1f>::iterator it = chars.begin(); it != chars.end(); ++it) {
		
		// compute feature vector
		cv::Moments moments = cv::moments(*it, true);
		double hu[7];
		cv::HuMoments(moments, hu);

		// save to struct
		CaptchaChar cc;
		cc.char_img = *it;
		cc.feature_vector = cv::Mat1f::zeros(1, 7);
		for (int i = 0; i < 7; ++i) {
			cc.feature_vector(0, i) = static_cast<float>(hu[i]);
		}
		captcha_.extracted_chars.push_back(cc);
	}

	if (verbose_) std::cout << "done." << std::endl;
}

void CaptchaProcessor::show_process_image(const cv::Mat1f &img) {

	cv::namedWindow("processing captcha", CV_WINDOW_NORMAL);
	cv::imshow("processing captcha", img);
	cv::waitKey(TIME_TO_WAIT);
}