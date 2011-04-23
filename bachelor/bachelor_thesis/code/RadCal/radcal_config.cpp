#include "radcal_config.h"

#ifdef VOLE_GUI
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#endif // VOLE_GI


namespace vole {

RadCalConfig::RadCalConfig(const std::string &prefix) : Config(prefix) {
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}


std::string RadCalConfig::getString() {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else { // no prefix, i.e. we handle I/O parameters
		s << "input=" << filenameImage << " # Image to process" << std::endl
		  << "output=" << filenameOutputImage << " # Working directory" << std::endl
		  << "outputInverseCRF=" << outputInverseCRF
		  	<< "# output textfile with estimated inverse CRFs for all three channels" << std::endl
		  << "root_dir=" << root_dir
		  	<< "# root directory for DoRF database files" << std::endl;
	}

	s << "maxIter=" << maxIter << " # maximum number of iterations for Levenberg-Marquardt routine" << std::endl
	<< "epsilonMax=" << epsilonMax << " # maximum color distance in a region" << std::endl
	<< "epsilonMin=" << epsilonMin << " # minimum color distance between two regions" << std::endl
	<< "patchSize=" << patchSize << " # size of examined patches" << std::endl
	<< "stepWidthX=" << stepWidthX << " # increment in x-direction (observation set formation speedup)" << std::endl
	<< "stepWidthY=" << stepWidthY << " # increment in y-direction (observation set formation speedup)" << std::endl
	<< "dilation=" << dilation << " # parameter for dilation" << std::endl
	<< "lambda=" << lambda
		<< " # setting for lambda, balances prior knowledge and image-specific information"
		<< std::endl
	<< "nKernels=" << nKernels << " # number of used kernels (kappa) for the Gaussian mixture model" << std::endl
	<< "nPCAcomponents=" << nPCAcomponents << " # number of used eigenvectors (PCA-representation)" << std::endl
	;

	return s.str();
}



#ifdef WITH_BOOST
void RadCalConfig::initBoostOptions() {
	options.add_options()
		(key("input,I"), value(&filenameImage)->default_value("img/test.png"),
			 "Image to process")
		(key("output,O"), value(&filenameOutputImage)->default_value("linearized.png"),
			 "CRF corrected output image")
		(key("outputInverseCRF"), value(&outputInverseCRF)->default_value("inverseCRFfilename.txt"),
		  	"output textfile with estimated inverse CRFs for all three channels")
		(key("root_dir"), value(&root_dir)->default_value("/disks/data1/riess/code/reflectance/crf/RadCal/", ".../code/.../RadCal/"),
		  	"root directory for DoRF database files")
		(key("maxIter"), value(&maxIter)->default_value(250),
			"maximum number of iterations for Levenberg-Marquardt routine")
		(key("epsilonMax"), value(&epsilonMax)->default_value(0.3, "0.3"),
		    "maximum color distance in a region")
		(key("epsilonMin"), value(&epsilonMin)->default_value(0.5, "0.5"),
			"minimum color distance between two regions")
		(key("patchSize"), value(&patchSize)->default_value(15),
			"size of examined patches")
		(key("stepWidthX"), value(&stepWidthX)->default_value(3),
			"increment in x-direction (observation set formation speedup)")
		(key("stepWidthY"), value(&stepWidthY)->default_value(3),
		    "increment in y-direction (observation set formation speedup)")
		(key("dilation"), value(&dilation)->default_value(7),
			"parameter for dilation")
		(key("lambda"), value(&lambda)->default_value(1000),
			"balances prior knowledge and image-specific information")
		(key("nKernels"), value(&nKernels)->default_value(5),
			"number of used kernels (kappa) for the Gaussian mixture model")
		(key("nPCAcomponents"), value(&nPCAcomponents)->default_value(5),
			"number of used eigenvectors (PCA-representation)")
	;
}
#endif // WITH_BOOST


#ifdef VOLE_GUI
QWidget *RadCalConfig::getConfigWidget() {
	this->initConfigWidget();
	QVBoxLayout *jpeg_config = new QVBoxLayout();
	jpeg_config->addWidget(new QLabel("FIXME"));
	layout->addLayout(jpeg_config);
	configWidget->setLayout(layout);
	return configWidget;
}

void RadCalConfig::updateValuesFromWidget() {
//	put something reasonable here once we have a graphical version
//	{ std::stringstream s; s << edit_sigma->text().toStdString();
//		s >> sigma; }
}
#endif // VOLE_GUI



}

