#include "ClosestPointFinderBruteForceGPU.h"

#include <iostream>

extern "C"
void initGPUBruteForce(PointCoords* targetCoords, PointColors* targetColors, int nrOfPoints);

extern "C"
void cleanupGPUBruteForce(); 

extern "C"
void FindClosestPointsGPUBruteForce(int nrOfPoints, int metric, bool useRGBData, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, float* distances);

unsigned short*
ClosestPointFinderBruteForceGPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	FindClosestPointsGPUBruteForce(m_NrOfPoints, m_Metric, m_UseRGBData, m_WeightRGB, m_Indices, sourceCoords, sourceColors, m_Distances);

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void ClosestPointFinderBruteForceGPU::SetTarget(PointCoords* targetCoords, PointColors* targetColors) 
{ 
	ClosestPointFinder::SetTarget(targetCoords, targetColors);
	initGPUBruteForce(targetCoords, targetColors, m_NrOfPoints);
}
