#ifndef ClosestPointFinderBruteForceGPU_H__
#define	ClosestPointFinderBruteForceGPU_H__

#include "ClosestPointFinder.h"


class ClosestPointFinderBruteForceGPU : public ClosestPointFinder
{
public:
	ClosestPointFinderBruteForceGPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints) { }
	~ClosestPointFinderBruteForceGPU();

	void SetTarget(Point6D* target); 
	int* FindClosestPoints(Point6D *source);
	inline bool usesGPU() { return true; }

};



#endif // ClosestPointFinderBruteForceGPU_H__
