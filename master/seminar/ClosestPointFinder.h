#ifndef ClosestPointFinder_H__
#define	ClosestPointFinder_H__

#include "defs.h"

class ClosestPointFinder {

public:
	ClosestPointFinder(int nrPoints) : m_NrOfPoints(nrPoints) 
	{
		m_Indices = new int[m_NrOfPoints];
	}

	virtual ~ClosestPointFinder() { delete[] m_Indices; }

	inline void SetTarget(Point6D* target) { m_Target = target; }
	inline void SetMetric(int metric) { m_Metric = metric; }
	virtual int* FindClosestPoints(Point6D* source) = 0;

protected:
	int m_NrOfPoints;
	int m_Metric;
	int* m_Indices; 
	Point6D* m_Target;

};

#endif // ClosestPointFinder_H__