#include "geoinv_config.h"

#ifdef VOLE_GUI
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#endif // VOLE_GI


namespace vole {

GeoInvConfig::GeoInvConfig(const std::string &prefix) : Config(prefix) {
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}


std::string GeoInvConfig::getString() {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else { // no prefix, i.e. we handle I/O parameters
		s << "input=" << filenameImage << " # Image to process" << std::endl
		  << "output=" << outputPath << " # Working directory" << std::endl
		  << "trainingFile=" << histogramsTraining << "# input training file for weighting" << std::endl;
	}

	s << "errorThreshold=" << errorThreshold << " # maximum error function value E(R)" << std::endl
	<< "channel=" << channel << " # channel used for computations (0,1,2 <-> B,G,R)" << std::endl
	<< "gamma=" << gamma << " # gamma curve parameter for gamma curve added before computations" << std::endl
	;

	return s.str();
}



#ifdef WITH_BOOST
void GeoInvConfig::initBoostOptions() {
	options.add_options()
		(key("input,I"), value(&filenameImage)->default_value("img/test.png"),
			 "Image to process")
		(key("output,O"), value(&outputPath)->default_value(""),
			 "output path for all computed data")
		(key("channel"), value(&channel)->default_value(0),
		  	"channel used for computations (0,1,2 <-> B,G,R)")
		(key("errorThreshold"), value(&errorThreshold)->default_value(10),
			"maximum error function value for E(R)")
		(key("gamma"), value(&gamma)->default_value(1),
			"gamma curve parameter for gamma curve added before computations")
		(key("histogramsTraining"), value(&histogramsTraining)->default_value("training/training.dat"),
			"training histograms used for weighting")
	;
}
#endif // WITH_BOOST


#ifdef VOLE_GUI
QWidget *GeoInvConfig::getConfigWidget() {
	this->initConfigWidget();
	QVBoxLayout *jpeg_config = new QVBoxLayout();
	jpeg_config->addWidget(new QLabel("FIXME"));
	layout->addLayout(jpeg_config);
	configWidget->setLayout(layout);
	return configWidget;
}

void GeoInvConfig::updateValuesFromWidget() {
//	put something reasonable here once we have a graphical version
//	{ std::stringstream s; s << edit_sigma->text().toStdString();
//		s >> sigma; }
}
#endif // VOLE_GUI



}

