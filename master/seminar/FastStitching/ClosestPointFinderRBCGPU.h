#ifndef ClosestPointFinderRBCGPU_H__
#define ClosestPointFinderRBCGPU_H__

#include "ClosestPointFinder.h"
#include <list>
#include <vector>
#include <algorithm>
#include <limits>


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
		unsigned int index;
		std::list<unsigned int> points;
	} Representative;

public:
	ClosestPointFinderRBCGPU(int NrOfPoints, int metric, int weightRGB, float nrOfRepsFactor);

	~ClosestPointFinderRBCGPU();

	void Initialize(float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors);

	void FindClosestPoints(int* indices, float* distances);


protected:
	void initRBC();

	int m_NrOfReps;
	float m_NrOfRepsFactor;
	bool m_Initialized;

	std::vector<Representative> m_Representatives;
	RepGPU* m_RepsGPU;
	unsigned int* m_Reps;
	unsigned int* m_PointToRep;
	unsigned int* m_RepsIndices;
	
};



#endif // ClosestPointFinderRBCGPU2_H__
