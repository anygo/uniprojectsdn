#ifndef ClosestPointFinder_H__
#define	ClosestPointFinder_H__

#include "defs.h"

class ClosestPointFinder
{

public:
	ClosestPointFinder(int nrPoints) : m_NrOfPoints(nrPoints), m_Indices(new unsigned short[nrPoints]) {}
	virtual ~ClosestPointFinder() { delete[] m_Indices; }

	virtual unsigned short* FindClosestPoints(Point6D* source) = 0;

	virtual inline void SetTarget(Point6D* target) { m_Target = target; }
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
	Point6D* m_Target;

};

#endif // ClosestPointFinder_H__