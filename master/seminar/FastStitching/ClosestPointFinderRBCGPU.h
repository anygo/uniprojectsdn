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
	ClosestPointFinderRBCGPU(int NrOfPoints, float weightRGB, float nrOfRepsFactor);

	~ClosestPointFinderRBCGPU();

	void Initialize(float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors);

	void FindClosestPoints(unsigned int* indices, float* distances);


protected:

	void InitializeRBC();

	int m_NrOfReps;
	float m_NrOfRepsFactor;
	bool m_Initialized;

	std::vector<Representative> m_Representatives;
	RepGPU* m_RepsGPU;
	unsigned int* m_RepIndices;
	unsigned int* m_PointToRep;
	
	// GPU pointer
	float* m_devDistances;
	unsigned int* m_devIndices;

	// RBC GPU pointer
	unsigned int* m_devRepIndices;
	unsigned int* m_devPointToRep;
	unsigned int* m_devReps;

};



#endif // ClosestPointFinderRBCGPU2_H__
