#ifndef ClosestPointFinderBruteForceGPU_H__
#define	ClosestPointFinderBruteForceGPU_H__

#include "ClosestPointFinder.h"


class ClosestPointFinderBruteForceGPU : public ClosestPointFinder
{
public:
	ClosestPointFinderBruteForceGPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints) { }
	~ClosestPointFinderBruteForceGPU();

	void SetTarget(PointCoords* targetCoords, PointColors* targetColors); 
	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	inline bool usesGPU() { return true; }

};



#endif // ClosestPointFinderBruteForceGPU_H__
