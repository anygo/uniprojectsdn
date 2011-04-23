#ifndef TRAINER_H
#define TRAINER_H

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "definitions.h"

class Trainer
{
public:
	Trainer();
	void reset();
	void setNumBins(int binsPerFeature[NUM_FEATURES]);
	void addSet(std::vector<LISO>& lisoSet);
	void computeHists(bool normalized);
	void saveResult(std::string filename);
	void printFeatures(std::string filename, std::vector<LISO>& lisoSet);
	void printHists(std::string prefix);
	void computeFeatures(std::vector<LISO>& lisoSet, cv::Mat1f& lisoMap);
	
	
	
protected:
	FeatureBoundary m_featureBoundaries[NUM_FEATURES];
	double *m_featureHistLPIP[NUM_FEATURES]; // using double instead of int (so we can normalize later)
	double *m_featureHistNonLPIP[NUM_FEATURES]; // using double instead of int (so we can normalize later)

	std::vector<LISO> m_lpip;
	std::vector<LISO> m_nonlpip;

	bool m_histComputed;
};


#endif // TRAINER_H