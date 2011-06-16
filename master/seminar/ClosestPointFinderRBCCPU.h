#ifndef ClosestPointFinderRBCCPU_H__
#define ClosestPointFinderRBCCPU_H__

#include "ClosestPointFinder.h"
#include <list>
#include <vector>


/**	@class		ClosestPointFinderRBCCPU
 *	@brief		RandomBallCover ClosestPointFinder on CPU
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that implements the ClosestPointFinder Interface by using
 *	the Random Ball Cover acceleration structure.
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
	ClosestPointFinderRBCCPU(int NrOfPoints, float nrOfRepsFactor) : ClosestPointFinder(NrOfPoints), m_NrOfRepsFactor(nrOfRepsFactor) { }

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	void SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors)
	{
		ClosestPointFinder::SetTarget(targetCoords, targetColors, sourceCoords, sourceColors);
		initRBC();
	}
	inline bool usesGPU() { return false; }

protected:
	void initRBC();
	float DistanceTargetTarget(unsigned short i, unsigned short j);
	float DistanceSourceTarget(PointCoords sourceCoords, PointColors sourceColors, unsigned short j);

	std::vector<Representative> m_Representatives;
	float m_NrOfRepsFactor;
};



#endif // ClosestPointFinderRBCCPU_H__
