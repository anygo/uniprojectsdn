#ifndef DECAPTCHATOOL_H
#define DECAPTCHATOOL_H

#include "decaptcha_tool.h"

void DecaptchaTool::add_image(const std::string &filename) {
	
	cv::Mat1f img = cv::imread(filename, 0) / 255.f; // grayscale intensities in [0; 1]
	Captcha captcha;
	captcha.orig_img = img.clone();
	captcha.processed_img = img.clone();

	captchas_.push_back(captcha);
}

void DecaptchaTool::test(const std::string &classifier_file_read) {
	
	classifier_.load(classifier_file_read.c_str());
	std::cout << classifier_file_read << " has been loaded successfully" << std::endl;

	for (std::vector<Captcha>::iterator it = captchas_.begin(); it != captchas_.end(); ++it) {
		
		cv::namedWindow("orig", CV_WINDOW_NORMAL);
		cv::imshow("orig", (*it).orig_img);

		CaptchaProcessor cp(*it, true);
		cp.process();

		std::cout << "prediction: ";
		for (std::vector<CaptchaChar>::iterator cc_it = (*it).extracted_chars.begin(); cc_it != (*it).extracted_chars.end(); ++cc_it) {
			
			char prediction;
			prediction = static_cast<char>(classifier_.predict((*cc_it).feature_vector));
			std::cout << prediction;
		}
		std::cout << std::endl;
		cv::waitKey(5000);
	}
}

void DecaptchaTool::train(const std::string &classifier_file_write) {
	
	// matrices for training (6 chars per captcha, 7 features)
	cv::Mat1f train_data(captchas_.size()*6, 7);
	cv::Mat1f train_classes(captchas_.size()*6, 1);

	int cur_row = 0;
	for (std::vector<Captcha>::iterator it = captchas_.begin(); it != captchas_.end(); ++it) {
		
		cv::namedWindow("orig", CV_WINDOW_NORMAL);
		cv::imshow("orig", (*it).orig_img);

		CaptchaProcessor cp(*it, true);
		cp.process();

		std::cout << "  => now label the single characters that show up! (focus must be on image)" << std::endl;
		for (std::vector<CaptchaChar>::iterator cc_it = (*it).extracted_chars.begin(); cc_it != (*it).extracted_chars.end(); ++cc_it) {
			
			cv::namedWindow("char to be labeled", CV_WINDOW_NORMAL);
			cv::imshow("char to be labeled", (*cc_it).char_img);

			char label = cv::waitKey();
			(*cc_it).label = label;

			for (int i = 0; i < 7; ++i) {
				train_data(cur_row, i) = (*cc_it).feature_vector(0, i);
			}
			//train_data.row(cur_row) = (*cc_it).feature_vector.clone();
			//(*cc_it).feature_vector.copyTo(train_data.row(cur_row));
			train_classes(cur_row, 0) = (*cc_it).label;

			++cur_row;
		}
	}

	std::cout << "training classifier..." << std::endl;
	classifier_.train(train_data, train_classes);

	std::cout << "saving trained classifier to " << classifier_file_write << std::endl;
	classifier_.save(classifier_file_write.c_str());
	std::cout << "done. you can now use it for testing!" << std::endl;
}


#endif // DECAPTCHATOOL_H
