#include "observation_set_formator.h"
#include "likelihood_function.h"
#include "prior_model.h"
#include "optimizer.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include "radcal_config.h"

using namespace cv;


int main(int argc, char *argv[])
{
	// simulation der Umgebung: Config-Objekt einigermassen sinnvoll fuellen
	RadCalConfig config;

	// step 1: data acquisition
	ObservationSetFormator osf(config.filenameImage, config.epsilonMax, config.epsilonMin, 
			config.patchSize, config.stepWidthX, config.stepWidthY, config.dilation);
	osf.formateObservationSet();	
	// compute beta (color coverage)
	std::vector<ColorTriple> omega = osf.getObservationSet();
	double coverage = osf.getCoverage();
	std::cout << "color coverage (beta): " << coverage << std::endl;
	// initialize likelihood function and prior model
	LikelihoodFunction lf("data/H.txt", "data/mean_curve.txt", config.lambda);
	PriorModel pm("data/invdorf_pca.txt", config.nKernels, config.nPCAcomponents);
	// step 2: crf estimation
	Optimizer opt(pm, lf, omega);
	opt.optimize(config.maxIter);

	// print function
	opt.printBestFunction(outputInverseCRF.c_str());

	// save image with applied crf
	cv::Mat3b img = imread(config.filenameImage);
	opt.applyBestFunction(img);
	imwrite(config.filenameOutputImage, img);

 	return 0;
}