#include "lpip_detector.h"
#include "crf_estimator.h"
#include "trainer.h"
#include "classifier.h"

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "math.h"
#include <iostream>


int main(int argc, char *argv[])
{	
	std::string filenameImage("small.png");
	double errorThreshold = 10.;
	int channel = 0;
	double gamma = 1.0;

	std::string histogramsTraining("training/training.dat");
	std::string outputPath("");


	// initialize LPIP detector
	LPIPDetector detector(filenameImage, channel, errorThreshold, 3, gamma);
	std::cout << "Detecting LISOs in the image..." << std::endl;
	detector.detect(outputPath);
	std::vector<LISO> lisoSet = detector.getLisoSet();
		
	// save LISO map 
	cv::Mat1f lisoMap = detector.getLisoMap();
	std::stringstream path1;
	path1 << outputPath << "liso_map.png";
	std::cout << "Save LISO map to " << path1.str() << std::endl;
	imwrite(path1.str(), lisoMap*255);

	// compute features and save histograms as images
	Trainer trainer;
	trainer.computeFeatures(lisoSet, lisoMap);
	std::cout << "Computing RQ Histogram..." << std::endl;
	Classifier classifier(histogramsTraining);
	classifier.createHistRQ(100, 50, lisoSet, false, "RQ_histogram");
	classifier.createHistRQ(100, 50, lisoSet, true, "RQ_histogram_weighted");

	
	return 0;
}