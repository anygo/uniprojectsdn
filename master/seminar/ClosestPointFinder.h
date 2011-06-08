#ifndef ClosestPointFinder_H__
#define	ClosestPointFinder_H__

#include "defs.h"

/**	@class		ClosestPointFinder
 *	@brief		Interface for all ClosestPointFinders
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Abstract class/Interface for the ClosestPointFinders implemented
 *  for the Stitching Plugin
 */
class ClosestPointFinder
{
public:
	ClosestPointFinder(int nrPoints) : m_NrOfPoints(nrPoints), m_Indices(new unsigned short[nrPoints]), m_Distances(new float[nrPoints]) {}
	virtual ~ClosestPointFinder() { delete[] m_Indices; delete[] m_Distances; }

	virtual unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors) = 0;

	virtual inline void SetTarget(PointCoords* targetCoords, PointColors* targetColors) { m_TargetCoords = targetCoords; m_TargetColors = targetColors; }
	virtual inline void SetUseRGBData(bool use) { m_UseRGBData = use; }
	virtual inline void SetWeightRGB(float weight) { m_WeightRGB = weight; }
	virtual inline void SetMetric(int metric) { m_Metric = metric; }
	virtual inline float* GetDistances() { return m_Distances; }
	virtual inline bool usesGPU() { return false; }


protected:
	int m_NrOfPoints;
	int m_Metric;
	unsigned short* m_Indices;
	bool m_UseRGBData;
	float m_WeightRGB;
	PointCoords* m_TargetCoords;
	PointColors* m_TargetColors;
	float* m_Distances;
};

#endif // ClosestPointFinder_H__