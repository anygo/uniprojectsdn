#include <iostream>

#include "command_radcal.h"

#include "RadCal/observation_set_formator.h"
#include "RadCal/likelihood_function.h"
#include "RadCal/prior_model.h"
#include "RadCal/optimizer.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <sstream>


using namespace boost::program_options;

// if we want an enum for configuration, we need to call this
// ENUM_MAGIC(boundcheck);

namespace vole {

RadCal::RadCal()
 : Command(
		"radcal",
		config,
		"Dominik Neumann",
		"sidoneum@stud.informatik.uni-erlangen.de")
{
}

RadCal::~RadCal() {
}

int RadCal::execute() {

	// step 1: data acquisition
	ObservationSetFormator osf(config.filenameImage, config.epsilonMax, config.epsilonMin, 
			config.patchSize, config.stepWidthX, config.stepWidthY, config.dilation);
	osf.formateObservationSet();	
	// compute beta (color coverage)
	std::vector<ColorTriple> omega = osf.getObservationSet();
	double coverage = osf.getCoverage();
	std::cout << "color coverage (beta): " << coverage << std::endl;
	// initialize likelihood function and prior model
	std::string h_txt = getFilePath("data/H.txt");
	std::string mean_curve = getFilePath("data/mean_curve.txt");
	LikelihoodFunction lf(h_txt.c_str(), mean_curve.c_str(), config.lambda);
	std::string invdorf = getFilePath("data/invdorf_pca.txt");
	PriorModel pm(invdorf.c_str(), config.nKernels, config.nPCAcomponents);
	// step 2: crf estimation
	Optimizer opt(pm, lf, omega);
	opt.optimize(config.maxIter);

	// print function
	opt.printBestFunction(config.outputInverseCRF.c_str());

	// save image with applied crf
	cv::Mat3b img = cv::imread(config.filenameImage);
	opt.applyBestFunction(img);
	imwrite(config.filenameOutputImage, img);

 	return 0;
}

std::string RadCal::getFilePath(std::string filename) {
	std::stringstream s;
	s << config.root_dir << "/" << filename;
	return s.str();
}

void RadCal::printShortHelp() const {
	std::cout << "Radiometric Calibration Analysis Suite" << std::endl;
}

void RadCal::printHelp() const {
	std::cout << "Radiometric Calibration Analysis Suite" << std::endl;
	std::cout << std::endl;
	std::cout << "Algorithm by Lin et al." << std::endl;
	std::cout << "At first, the image-specific observation set gets formed." << std::endl;
	std::cout << "Then the color coverage factor (beta) gets computed and printed to stdout." << std::endl;
	std::cout << "Note that the prior model and likelihood function require three files:" << std::endl;
	std::cout << "  1) <root_dir>/data/H.txt" << std::endl;
	std::cout << "  2) <root_dir>/data/mean_curve.txt" << std::endl;
	std::cout << "  3) <root_dir>/data/invdorf_pca.txt" << std::endl << std::endl;
	std::cout << "Next step is Levenberg-Marquardt optimization (maxIter iterations)." << std::endl;
	std::cout << "The printed norm (stdout) is NOT the one from my thesis (\Psi)!" << std::endl;
	std::cout << "Result from LM-routine (successful | timed out (change maxIter!) | tolerable)" << std::endl << std::endl;
	std::cout << "Estimated inverse CRF is written to <outputInverseCRF>, where" << std::endl;
	std::cout << "  first column: x-axis; second to fourth column: channel -> B G R" << std::endl;
	std::cout << "  (e.g. if you want to plot the estimated CRFs)" << std::endl << std::endl;
	std::cout << "Resulting 'linear' image is written to <filenameOutputImage>" << std::endl << std::endl;
}

}
