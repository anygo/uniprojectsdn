#include <vector>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"
#include "captcha_processor.h"
#include "structs.h"

class DecaptchaTool {

public:

	DecaptchaTool() {}
	~DecaptchaTool() {}
	
	void add_image(const std::string &filename);
	void train(const std::string &classifier_file_write);
	void test(const std::string &classifier_file_read);

	
protected:
	//std::vector<cv::Mat1f> images_;
	cv::NormalBayesClassifier classifier_;
	std::vector<Captcha> captchas_;
};