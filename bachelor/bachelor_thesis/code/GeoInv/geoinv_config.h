#ifndef GEOINV_CONFIG_H
#define GEOINV_CONFIG_H

#include "config.h"

#include "cv.h"

#include <iostream>
#include <vector>

namespace vole {


class GeoInvConfig : public vole::Config {
	public:

	GeoInvConfig(const std::string &prefix = std::string());

	// input image filename
	std::string filenameImage;
	// output path for computed data
	std::string outputPath;
	// the channel to be examined (0,1,2 <-> B,G,R)
	int channel;
	// the maximum error function value (E(R))
	double errorThreshold;
	// gamma value, which is added (for linear images only)
	double gamma;
	// training file used for weighting
	std::string histogramsTraining;




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

#endif // GEOINV_CONFIG_H
