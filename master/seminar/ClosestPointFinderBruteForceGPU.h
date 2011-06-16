#ifndef ClosestPointFinderBruteForceGPU_H__
#define	ClosestPointFinderBruteForceGPU_H__

#include "ClosestPointFinder.h"

extern "C"
void cleanupGPUCommon(); 

/**	@class		ClosestPointFinderBruteForceGPU
 *	@brief		BruteForce ClosestPointFinder on GPU (CUDA)
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that implements the ClosestPointFinder and tries all combinations of
 *	points to find the closest points in two point clouds. Massively parallelized
 *	using a CUDA graphics card
 */
class ClosestPointFinderBruteForceGPU : public ClosestPointFinder
{
public:
	ClosestPointFinderBruteForceGPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints) { }
	~ClosestPointFinderBruteForceGPU() { cleanupGPUCommon(); }

	void SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors); 
	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	inline bool usesGPU() { return true; }

};



#endif // ClosestPointFinderBruteForceGPU_H__
