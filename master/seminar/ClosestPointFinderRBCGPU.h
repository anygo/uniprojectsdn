#ifndef ClosestPointFinderRBCGPU_H__
#define ClosestPointFinderRBCGPU_H__

#include "ClosestPointFinder.h"


extern "C"
void cleanupGPURBC(); 

/**	@class		ClosestPointFinderRBCGPU
 *	@brief		
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	
 */
class ClosestPointFinderRBCGPU : public ClosestPointFinder
{

public:
	ClosestPointFinderRBCGPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints), m_PointToRep(new unsigned short[NrOfPoints]) { }
	~ClosestPointFinderRBCGPU() { delete[] m_Representatives; delete[] m_PointToRep; cleanupGPURBC(); }

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	void SetTarget(PointCoords* targetCoords, PointColors* targetColors) { ClosestPointFinder::SetTarget(targetCoords, targetColors); initRBC(); }
	inline bool usesGPU() { return true; }

protected:
	void initRBC();
	float DistanceTargetTarget(unsigned short i, unsigned short j);

	unsigned short* m_Representatives;
	unsigned short* m_PointToRep;
	
};



#endif // ClosestPointFinderRBCGPU_H__
