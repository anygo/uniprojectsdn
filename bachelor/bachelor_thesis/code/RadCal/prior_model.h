#ifndef PRIOR_MODEL_H
#define PRIOR_MODEL_H

#include "cv.h"
#include "cxcore.h"
#include "ml.h"
#include "highgui.h"

class PriorModel
{
public:
	PriorModel(const char *filenameInverseResponseFunctionsPCA, int nKernels = 5, int nPCAcomponents = 5);
	double prior(cv::Mat1f &coefficients);
	inline int getNPCAComponents() { return m_nPCAComponents ;}

	//should no be needed
	inline cv::Mat1f getCoefficientsFromDoRF(int index) { return m_samples.row(index); }

protected:
	double NormalDistribution(cv::Mat1f &cov, cv::Mat1f mean, cv::Mat1f x);
	void setEMParams();
	cv::Mat1f readMatrix(const char *filename);

	CvEM m_EMmodel;
	CvEMParams m_EMparams;
	cv::Mat1f m_samples;
	int m_nKernels;
	int m_nPCAComponents;
};

#endif
