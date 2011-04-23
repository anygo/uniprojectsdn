#ifndef LPIP_DETECTOR_H
#define LPIP_DETECTOR_H

#include "definitions.h"

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

class LPIPDetector
{
public:
	// constructor
	LPIPDetector(const std::string filename, unsigned int channel, double errorThreshold = 10., int sobelSize = 3, double gamma = 1.);
	// main routine
	void detect(std::string outputpath);
	inline std::vector<LISO>& getLisoSet() { return m_lisoSet; }
	inline cv::Mat1f& getLisoMap() { return m_lisoMap; }

protected:
	// helper functions
	double gaussFunction(double S, double sigma);
	void getDerivativesAfterRotation(cv::Mat1f &img, double degree, Derivatives &d);

	// member variables
	cv::Mat1f m_coi;
	std::vector<LISO> m_lisoSet;
	double m_errorThreshold;
	cv::Mat1f m_lisoMap;
	int m_sobelSize;
	std::string m_filename;
	int m_channel;
	double m_gamma;
};


#endif // LPIP_DETECTOR_H