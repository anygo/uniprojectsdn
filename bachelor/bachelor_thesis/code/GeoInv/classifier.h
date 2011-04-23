#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "definitions.h"

class Classifier
{
public:
	Classifier(std::string filename);
	void load(std::string filename);
	double probabilityFunction(double featureVector[NUM_FEATURES]);
	void createHistRQ(int binsX, int binsY, std::vector<LISO>& lisoSet, bool weighted, std::string title);
	
protected:
	double m_PLPIP; // P(LPIP)
	double m_PNonLPIP; // P(NonLPIP)

	FeatureBoundary m_featureBoundaries[NUM_FEATURES];
	double *m_featureHistLPIP[NUM_FEATURES]; 
	double *m_featureHistNonLPIP[NUM_FEATURES]; 
	
};


#endif // CLASSIFIER_H