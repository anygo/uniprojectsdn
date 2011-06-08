#ifndef ClosestPointFinderRBCCPU_H__
#define ClosestPointFinderRBCCPU_H__

#include "ClosestPointFinder.h"
#include <list>
#include <vector>


/**	@class		ClosestPointFinderRBCCPU
 *	@brief		
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	
 */
class ClosestPointFinderRBCCPU : public ClosestPointFinder
{

	typedef struct Representative
	{
		unsigned short index;
		std::list<unsigned short> points;
		float radius;
	} Representative;


public:
	ClosestPointFinderRBCCPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints) { }

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	void SetTarget(PointCoords* targetCoords, PointColors* targetColors) { ClosestPointFinder::SetTarget(targetCoords, targetColors); initRBC(); }

protected:
	void initRBC();
	float DistanceTargetTarget(unsigned short i, unsigned short j);
	float DistanceSourceTarget(PointCoords sourceCoords, PointColors sourceColors, unsigned short j);

	std::vector<Representative> m_Representatives;
	
};



#endif // ClosestPointFinderRBCCPU_H__
