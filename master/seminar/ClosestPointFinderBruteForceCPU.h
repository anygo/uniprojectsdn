#ifndef ClosestPointFinderBruteForceCPU_H__
#define	ClosestPointFinderBruteForceCPU_H__

#include "ClosestPointFinder.h"


class ClosestPointFinderBruteForceCPU :
	public ClosestPointFinder
{

public:
	ClosestPointFinderBruteForceCPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints) { }

	int* FindClosestPoints(Point6D *source);
	int FindClosestPoint(Point6D point);

};

#endif //ClosestPointFinderBruteForceCPU_H__
