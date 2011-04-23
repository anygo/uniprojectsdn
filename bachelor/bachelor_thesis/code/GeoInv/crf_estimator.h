#ifndef CRF_ESTIMATOR_H
#define CRF_ESTIMATOR_H

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "definitions.h"
#include "classifier.h"

#define BINS_FOR_R 64

class CRFEstimator
{
public:
	CRFEstimator(Classifier& classifier, std::vector<LISO>& lisoSet);
	void estimate();
	double functionQ(double R, double alpha0, double alpha1, bool calibrated); // equation 22
	double objectiveFunction(double alpha0, double alpha1); // equation 24
	void plot(double alpha0, double alpha1);

	

protected:
	Classifier *m_classifier;
	std::vector<LISO> m_lisoSet;
	double m_distributionR[BINS_FOR_R];
};


#endif // CRF_ESTIMATOR_H