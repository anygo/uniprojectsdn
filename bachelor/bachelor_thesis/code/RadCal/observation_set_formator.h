#ifndef OBSERVATION_SET_FORMATOR_H
#define OBSERVATION_SET_FORMATOR_H

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "color_triple.h"
#include <iostream>

typedef struct PIXEL_INFO
{
	cv::Point coords;
	cv::Vec3f color;
} PIXEL_INFO;


class ObservationSetFormator
{
public:
	// constructor
	ObservationSetFormator(
			const std::string &filename, 
			double maxDistanceInRegion = 0.3, 
			double minDistanceBetweenRegions = 0.4, 
			int patchsize = 15,
			int stepWidthX = 1,
			int stepWidthY = 1,
			int dilation = 7, 
			double cannyThreshold1 = 20.,
			double cannyThreshold2 = 50.,
			int cannyApertureSize = 3
			) :
			m_maxDistanceInRegion(maxDistanceInRegion),
			m_minDistanceBetweenRegions(minDistanceBetweenRegions),
			m_patchsize(patchsize),
			m_stepWidthX(stepWidthX),
			m_stepWidthY(stepWidthY),
			m_cannyThreshold1(cannyThreshold1),
			m_cannyThreshold2(cannyThreshold2),
			m_cannyApertureSize(cannyApertureSize)
		{
			m_dilateKernel = cv::Mat1b::ones(dilation, dilation);
			m_image = cv::imread(filename);
			m_image *= (1./255.);
		}

	// main routine
	void formateObservationSet();
	inline std::vector<ColorTriple> getObservationSet() { return m_observationSet; }
	// computes the coverage factor (beta)
	double getCoverage();
	// not really used (just for evaluation)
	std::vector<ColorTriple> generateSyntheticSet(double gamma);


protected:
	void createEdgeImage(cv::Mat3f &src, cv::Mat1f &dst);
	bool checkPatch(const cv::Rect &r);
	cv::Point scanBorder(cv::Mat1f &edgePatch);
	void getMeanColor(std::vector<PIXEL_INFO> &vector, 
		cv::Mat1f &edgePatch, 
		cv::Mat3f &imagePatch,
		cv::Point start, // start point for flooding
		bool white // true -> Mp; false -> M1 | M2
		); 
	bool checkAndAddColorTriple(ColorTriple &colorTriple);

	// member variables
	cv::Mat3f m_image;
	cv::Mat1f m_edges;
	cv::Mat1b m_dilateKernel;
	
	// variables for iterating through the image
	int m_stepWidthX;
	int m_stepWidthY;
	int m_patchsize;

	// conditions, which have to be met by particular ColorTriple before
	// adding to the ObservationSet
	double m_maxDistanceInRegion;
	double m_minDistanceBetweenRegions;

	// the ObservationSet
	std::vector<ColorTriple> m_observationSet;

	// variables for creating the edge image
	double m_cannyThreshold1;
	double m_cannyThreshold2;
	int m_cannyApertureSize;
	bool m_cannyL2Gradient;
	
};

#endif
