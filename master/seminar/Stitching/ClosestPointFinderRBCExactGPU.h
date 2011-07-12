#ifndef ClosestPointFinderRBCExactGPU_H__
#define ClosestPointFinderRBCExactGPU_H__

#include "ClosestPointFinder.h"
#include <list>
#include <vector>
#include <algorithm>
#include <limits>


/**	@class		ClosestPointFinderRBCExactGPU
 *	@brief		RandomBallCover ClosestPointFinder on GPU (CUDA)
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that implements the ClosestPointFinder Interface by using
 *	the Random Ball Cover acceleration structure. Uses GPU capabilities.
 */
class ClosestPointFinderRBCExactGPU : public ClosestPointFinder
{

	typedef struct Representative
	{
		unsigned short index;
		std::list<unsigned short> points;
	} Representative;

public:
	ClosestPointFinderRBCExactGPU(int NrOfPoints, float nrOfRepsFactor);

	~ClosestPointFinderRBCExactGPU();

	void SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors);

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);

	inline bool usesGPU() { return true; }

protected:
	void initRBC();

	int m_NrOfReps;
	float m_NrOfRepsFactor;
	bool m_Initialized;

	std::vector<Representative> m_Representatives;
	RepGPU* m_RepsGPU;
	unsigned short* m_Reps;
	unsigned short* m_PointToRep;
	unsigned short* m_RepsIndices;
	
};



#endif // ClosestPointFinderRBCGPU2_H__
