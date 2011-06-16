#include "ClosestPointFinderBruteForceGPU.h"

#include <iostream>

extern "C"
void initGPUCommon(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors, float weightRGB, int metric, int nrOfPoints);

extern "C"
void FindClosestPointsGPUBruteForce(unsigned short* indices, float* distances);

unsigned short*
ClosestPointFinderBruteForceGPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	FindClosestPointsGPUBruteForce(m_Indices, m_Distances);

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void ClosestPointFinderBruteForceGPU::SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors) 
{ 
	ClosestPointFinder::SetTarget(targetCoords, targetColors, sourceCoords, sourceColors);
	initGPUCommon(m_TargetCoords, m_TargetColors, m_SourceCoords, m_SourceColors, m_WeightRGB, m_Metric, m_NrOfPoints);
}
