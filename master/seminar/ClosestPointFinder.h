#ifndef ClosestPointFinder_H__
#define	ClosestPointFinder_H__

#include "defs.h"

class ClosestPointFinder
{

public:
	ClosestPointFinder(int nrPoints) : m_NrOfPoints(nrPoints), m_Indices(new unsigned short[nrPoints]) {}
	virtual ~ClosestPointFinder() { delete[] m_Indices; }

	virtual unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors) = 0;

	virtual inline void SetTarget(PointCoords* targetCoords, PointColors* targetColors) { m_TargetCoords = targetCoords; m_TargetColors = targetColors; }
	virtual inline void SetUseRGBData(bool use) { m_UseRGBData = use; }
	virtual inline void SetWeightRGB(double weight) { m_WeightRGB = weight; }
	virtual inline void SetMetric(int metric) { m_Metric = metric; }
	virtual inline bool usesGPU() { return false; }


protected:
	int m_NrOfPoints;
	int m_Metric;
	unsigned short* m_Indices;
	bool m_UseRGBData;
	double m_WeightRGB;
	PointCoords* m_TargetCoords;
	PointColors* m_TargetColors;

};

#endif // ClosestPointFinder_H__