#include "ClosestPointFinderBruteForceGPU.h"

#include <iostream>

extern "C"
void initGPU(PointCoords* targetCoords, PointColors* targetColors, int nrOfPoints);

extern "C"
void cleanupGPU(); 

extern "C"
void FindClosestPointsCUDA(int nrOfPoints, int metric, bool useRGBData, double weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors);

unsigned short*
ClosestPointFinderBruteForceGPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	FindClosestPointsCUDA(m_NrOfPoints, m_Metric, m_UseRGBData, m_WeightRGB, m_Indices, sourceCoords, sourceColors);

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void ClosestPointFinderBruteForceGPU::SetTarget(PointCoords* targetCoords, PointColors* targetColors) 
{ 
	
	ClosestPointFinder::SetTarget( targetCoords, targetColors);
	initGPU(targetCoords, targetColors, m_NrOfPoints);
}

ClosestPointFinderBruteForceGPU::~ClosestPointFinderBruteForceGPU()
{ 
	cleanupGPU();
}