#ifndef ClosestPointFinderRBCGPU_H__
#define ClosestPointFinderRBCGPU_H__

#include "ClosestPointFinder.h"
#include <list>
#include <vector>


/**	@class		ClosestPointFinderRBCCPU
 *	@brief		RandomBallCover ClosestPointFinder on GPU (CUDA)
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that implements the ClosestPointFinder Interface by using
 *	the Random Ball Cover acceleration structure. Uses GPU capabilities.
 */
class ClosestPointFinderRBCGPU : public ClosestPointFinder
{

	typedef struct Representative
	{
		unsigned short index;
		std::list<unsigned short> points;
	} Representative;

public:
	ClosestPointFinderRBCGPU(int NrOfPoints, float nrOfRepsFactor) : ClosestPointFinder(NrOfPoints), m_NrOfRepsFactor(nrOfRepsFactor) { }
	~ClosestPointFinderRBCGPU();

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	void SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors) {
		ClosestPointFinder::SetTarget(targetCoords, targetColors, sourceCoords, sourceColors);
		initRBC();
	}
	inline bool usesGPU() { return true; }

protected:
	void initRBC();
	float DistanceTargetTarget(unsigned short i, unsigned short j);

	int m_NrOfReps;
	std::vector<Representative> m_Representatives;
	RepGPU* m_RepsGPU;
	float m_NrOfRepsFactor;
};



#endif // ClosestPointFinderRBCGPU2_H__
