#ifndef ClosestPointFinderRBCGPU2_H__
#define ClosestPointFinderRBCGPU2_H__

#include "ClosestPointFinder.h"
#include <list>
#include <vector>


/**	@class		ClosestPointFinderRBCGPU2
 *	@brief		
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	
 */
class ClosestPointFinderRBCGPU2 : public ClosestPointFinder
{

	typedef struct Representative
	{
		unsigned short index;
		std::list<unsigned short> points;
	} Representative;

public:
	ClosestPointFinderRBCGPU2(int NrOfPoints, double nrOfRepsFactor) : ClosestPointFinder(NrOfPoints), m_NrOfRepsFactor(nrOfRepsFactor) { }
	~ClosestPointFinderRBCGPU2();

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	void SetTarget(PointCoords* targetCoords, PointColors* targetColors) { ClosestPointFinder::SetTarget(targetCoords, targetColors); initRBC(); }
	inline bool usesGPU() { return true; }

protected:
	void initRBC();
	float DistanceTargetTarget(unsigned short i, unsigned short j);

	int m_NrOfReps;
	std::vector<Representative> m_Representatives;
	RepGPU* m_RepsGPU;
	double m_NrOfRepsFactor;
};



#endif // ClosestPointFinderRBCGPU2_H__
