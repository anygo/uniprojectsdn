#ifndef ClosestPointFinder_H__
#define	ClosestPointFinder_H__

#include "defs.h"
#include <iostream>

/**	@class		ClosestPointFinder
 *	@brief		Interface for all ClosestPointFinders
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Abstract class/Interface for the ClosestPointFinders implemented
 *  for the FastStitching Plugin
 */
class ClosestPointFinder
{
public:
	ClosestPointFinder(int nrPoints, int metric, float weightRGB) : m_NrOfPoints(nrPoints), m_Metric(metric), m_WeightRGB(weightRGB) {}
	virtual ~ClosestPointFinder() {}

	virtual void FindClosestPoints(unsigned int* indices, unsigned int* distances) = 0;

	virtual inline void Initialize(float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors) 
	{
		m_devTargetCoords = targetCoords; 
		m_devTargetColors = targetColors;
		m_devSourceCoords = sourceCoords;
		m_devSourceColors = sourceColors;
	}

protected:

	int m_NrOfPoints;
	int m_Metric;
	float m_WeightRGB;

	float4* m_devTargetCoords;
	float4* m_devTargetColors;
	float4* m_devSourceCoords;
	float4* m_devSourceColors;
};

#endif // ClosestPointFinder_H__