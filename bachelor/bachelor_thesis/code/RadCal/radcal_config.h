#ifndef RADCAL_CONFIG_H
#define RADCAL_CONFIG_H

#include "config.h"

#include "cv.h"

#include <iostream>
#include <vector>

namespace vole {

/**
 * Configuration parameters for the jpeg package, to be
 * included by all classes that need to access it.
 */
class RadCalConfig : public vole::Config {
	public:

	RadCalConfig(const std::string &prefix = std::string());

	// input image filename
	std::string filenameImage;
	// output image filename (after inverse CRF transformation)
	std::string filenameOutputImage;
	// output textfile with estimated inverse CRFs for all three channels
	std::string outputInverseCRF;
	// root directory for DoRF database files
	std::string root_dir;
	// maximum number of iterations for Levenberg-Marquardt routine
	int maxIter;
	// maximum color distance in a region
	double epsilonMax;
	// minimum color distance between two regions
	double epsilonMin;
	// size of examined patches
	int patchSize;
	// increment in x-direction (observation set formation speedup)
	int stepWidthX;
	// increment in y-direction (observation set formation speedup)
	int stepWidthY;
	// parameter for dilation
	int dilation;
	// setting for lambda, balances prior knowledge and image-specific information
	double lambda;
	// number of used kernels (kappa) for the Gaussian mixture model
	int nKernels;
	// number of used eigenvectors (PCA-representation)
	int nPCAcomponents;



	virtual std::string getString();

	#ifdef VOLE_GUI
		virtual QWidget *getConfigWidget();
		virtual void updateValuesFromWidget();
	#endif// VOLE_GUI


	protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

	#ifdef VOLE_GUI
		QLineEdit *edit_sigma, *edit_k_threshold, *edit_min_size;
		QCheckBox *chk_chroma_img;
	#endif // VOLE_GUI

};

}

#endif // RADCAL_CONFIG_H
