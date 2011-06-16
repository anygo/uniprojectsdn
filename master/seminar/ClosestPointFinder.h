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

	virtual inline void SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors) 
	{
		m_TargetCoords = targetCoords; 
		m_TargetColors = targetColors;
		m_SourceCoords = sourceCoords;
		m_SourceColors = sourceColors;
	}

	virtual inline void SetWeightRGB(float weight) { m_WeightRGB = weight; }
	virtual inline void SetMetric(int metric) { m_Metric = metric; }
	virtual inline float* GetDistances() { return m_Distances; }
	virtual inline bool usesGPU() = 0;


protected:
	int m_NrOfPoints;
	int m_Metric;
	unsigned short* m_Indices;
	float m_WeightRGB;
	PointCoords* m_TargetCoords;
	PointColors* m_TargetColors;
	PointCoords* m_SourceCoords;
	PointColors* m_SourceColors;
	float* m_Distances;
};

#endif // ClosestPointFinder_H__