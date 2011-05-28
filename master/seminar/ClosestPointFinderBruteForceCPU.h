#ifndef ClosestPointFinderBruteForceCPU_H__
#define	ClosestPointFinderBruteForceCPU_H__

#include "ClosestPointFinder.h"


class ClosestPointFinderBruteForceCPU : public ClosestPointFinder
{
public:
	ClosestPointFinderBruteForceCPU(int NrOfPoints) : ClosestPointFinder(NrOfPoints) { }

	int* FindClosestPoints(Point6D *source);
	int FindClosestPoint(Point6D point);
};


// for the multithreaded version
class ClosestPointFinderBruteForceCPUWorker
{
public:
	ClosestPointFinderBruteForceCPUWorker(int from,
										  int to,
										  int nrOfPoints,
										  int metric,
										  bool useRGBData,
										  double weightRGB,
										  int* indices,
										  Point6D* source,
										  Point6D* target
										  ) 
										  : m_From(from),
										  m_To(to),
										  m_NrOfPoints(nrOfPoints),
										  m_Metric(metric),
										  m_UseRGBData(useRGBData),
										  m_WeightRGB(weightRGB),
										  m_Indices(indices),
										  m_Source(source),
										  m_Target(target)
										  {}
	~ClosestPointFinderBruteForceCPUWorker() {}

	void operator()();

protected:
	int m_From;
	int m_To;
	int m_NrOfPoints;
	int m_Metric;
	bool m_UseRGBData;
	double m_WeightRGB;
	int* m_Indices;
	Point6D* m_Source;
	Point6D* m_Target;
};

#endif //ClosestPointFinderBruteForceCPU_H__
