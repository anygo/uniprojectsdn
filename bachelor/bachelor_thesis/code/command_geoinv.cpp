#include <iostream>

#include "command_geoinv.h"

#include "GeoInv/lpip_detector.h"
#include "GeoInv/crf_estimator.h"
#include "GeoInv/trainer.h"
#include "GeoInv/classifier.h"

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "math.h"
#include <iostream>

using namespace boost::program_options;

// if we want an enum for configuration, we need to call this
// ENUM_MAGIC(boundcheck);

namespace vole {

GeoInv::GeoInv()
 : Command(
		"geoinv",
		config,
		"Dominik Neumann",
		"sidoneum@stud.informatik.uni-erlangen.de")
{
}

GeoInv::~GeoInv() {
}

int GeoInv::execute() {

	// initialize LPIP detector
	LPIPDetector detector(config.filenameImage, config.channel, config.errorThreshold, 3, config.gamma);
	std::cout << "Detecting LISOs in the image..." << std::endl;
	detector.detect(config.outputPath);
	std::vector<LISO> lisoSet = detector.getLisoSet();
		
	// save LISO map 
	cv::Mat1f lisoMap = detector.getLisoMap();
	std::stringstream path1;
	path1 << config.outputPath << "liso_map.png";
	std::cout << "Save LISO map to " << path1.str() << std::endl;
	imwrite(path1.str(), lisoMap*255);

	// compute features and save histograms as images
	Trainer trainer;
	trainer.computeFeatures(lisoSet, lisoMap);
	std::cout << "Computing QR Histogram..." << std::endl;
	Classifier classifier(config.histogramsTraining);
	classifier.createHistRQ(100, 50, lisoSet, false, "QR_histogram");
	classifier.createHistRQ(100, 50, lisoSet, true, "QR_histogram_weighted");

 	return 0;
}


void GeoInv::printShortHelp() const {
	std::cout << "Geometry Invariants Analysis Suite" << std::endl;
}

void GeoInv::printHelp() const {
	std::cout << "Geometry Invariants Analysis Suite" << std::endl;
	std::cout << std::endl;
	std::cout << "This tool does not cover the whole algorithm by Ng et al. (Problems)" << std::endl;
	std::cout << "The LISO-map for the input image is generated and written to liso_map.png" << std::endl;
	std::cout << "The QR-histograms (weighted and unweighted) get computed and saved in two ways:" << std::endl;
	std::cout << "  1) png image" << std::endl;
	std::cout << "  2) text file: first col: R, second col: Q, third row: count (normalized)" << std::endl << std::endl;
}

}
