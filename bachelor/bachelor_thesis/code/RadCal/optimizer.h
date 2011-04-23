#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#include "color_triple.h"
#include "prior_model.h"
#include "likelihood_function.h"


class Optimizer
{
public: 
	// constructor
	Optimizer(PriorModel &pm, LikelihoodFunction &lf, const std::vector<ColorTriple> &omega);
	// objective function (see paper)
	double objectiveFuntion(cv::Mat1f &coefficients);
	// LM kickoff
	std::string optimize(int maxIter);
	// prints functions to textfiles
	void printFunction(const char *filename, cv::Mat1f function);
	// plots functions in cv::namedWindow (buggy)
	void plotFunction(cv::Mat1f function);
	// estimated CRF applied to an image
	void applyBestFunction(cv::Mat3b &image);
	// prints the estimated function after optimization
	void printBestFunction(const char *filename);	
	// returns estimated CRF
	cv::Mat1f getBestFunction() { return m_bestFunction; }

	PriorModel *m_pm;
	LikelihoodFunction *m_lf;


protected:
	// the set of color triples (omega)
	const std::vector<ColorTriple> *m_omega;
	// gets computed during optimization
	cv::Mat1f m_bestFunction;
	// estimated coefficients of PCA representation
	cv::Mat1f m_bestCoefficients;
};


#endif