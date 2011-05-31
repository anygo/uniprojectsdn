#ifndef ClosestPointFinderBruteForceCPU_H__
#define	ClosestPointFinderBruteForceCPU_H__

#include "ClosestPointFinder.h"
#include "QThread.h"


class ClosestPointFinderBruteForceCPU : public ClosestPointFinder
{
public:
	ClosestPointFinderBruteForceCPU(int NrOfPoints, bool multithreaded) : m_Multithreaded(multithreaded), ClosestPointFinder(NrOfPoints) { }

	unsigned short* FindClosestPoints(Point6D *source);

protected:
	bool m_Multithreaded;
};

class ClosestPointFinderBruteForceCPUWorker : public QThread
{
public:
	ClosestPointFinderBruteForceCPUWorker() : QThread() {}
	~ClosestPointFinderBruteForceCPUWorker() {}

	void setConfig(	int from,
					int to,
					int nrOfPoints,
					int metric,
					bool useRGBData,
					double weightRGB,
					unsigned short* indices,
					Point6D* source,
					Point6D* target
				  ) 
	{
		m_From = from;
		m_To = to;
		m_NrOfPoints = nrOfPoints;
		m_Metric = metric;
		m_UseRGBData = useRGBData;
		m_WeightRGB = weightRGB;
		m_Indices = indices;
		m_Source = source;
		m_Target = target;
	}

protected:
	void run();

	int m_From;
	int m_To;
	int m_NrOfPoints;
	int m_Metric;
	bool m_UseRGBData;
	double m_WeightRGB;
	unsigned short* m_Indices;
	Point6D* m_Source;
	Point6D* m_Target;
};


#endif // ClosestPointFinderBruteForceCPU_H__
