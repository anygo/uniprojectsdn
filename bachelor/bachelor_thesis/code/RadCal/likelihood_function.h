#ifndef LIKELIHOOD_FUNCTION_H
#define LIKELIHOOD_FUNCTION_H

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "color_triple.h"

class LikelihoodFunction
{
public: 
	LikelihoodFunction(const char *filenameH, const char *filenameMeanCurve, double lambda);
	double getTotalDistance(const std::vector<ColorTriple> &omega, cv::Mat1f &coefficients);
	double likelihoodFunction(const std::vector<ColorTriple> &omega, cv::Mat1f &coefficients);
	inline double getLambda() { return m_lambda; }
	cv::Mat1f computeFunctions(cv::Mat1f &coefficients);

protected:
	cv::Mat1f readMatrix(const char *filename);

	// member variables (see equation 10)
	cv::Mat1f m_H;
	cv::Mat1f m_g0;
	double m_lambda;
};

#endif