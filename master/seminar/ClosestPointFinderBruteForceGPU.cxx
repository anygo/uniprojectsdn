#include "ClosestPointFinderBruteForceGPU.h"

#include <iostream>

extern "C"
void initGPU(Point6D* target, int nrOfPoints);

extern "C"
void cleanupGPU(); 

extern "C"
void FindClosestPointsCUDA(int nrOfPoints, int metric, bool useRGBData, double weightRGB, unsigned short* indices, Point6D* source);

unsigned short*
ClosestPointFinderBruteForceGPU::FindClosestPoints(Point6D *source)
{
	FindClosestPointsCUDA(m_NrOfPoints, m_Metric, m_UseRGBData, m_WeightRGB, m_Indices, source);

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void ClosestPointFinderBruteForceGPU::SetTarget(Point6D* target) 
{ 
	
	m_Target = target;
	initGPU(target, m_NrOfPoints);
}

ClosestPointFinderBruteForceGPU::~ClosestPointFinderBruteForceGPU()
{ 
	cleanupGPU();
}