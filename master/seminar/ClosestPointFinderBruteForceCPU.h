#ifndef ClosestPointFinderBruteForceCPU_H__
#define	ClosestPointFinderBruteForceCPU_H__

#include "ClosestPointFinder.h"
#include "QThread.h"

/**	@class		ClosestPointFinderBruteForceCPU
 *	@brief		BruteForce ClosestPointFinder on CPU (multithreaded)
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that implements the ClosestPointFinder and tries all combinations of
 *	points to find the closest points in two point clouds. Ability to split
 *	the task into many parts and process them parallely on different threads.
 */
class ClosestPointFinderBruteForceCPU : public ClosestPointFinder
{
public:
	ClosestPointFinderBruteForceCPU(int NrOfPoints, bool multithreaded) : m_Multithreaded(multithreaded), ClosestPointFinder(NrOfPoints) { }

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);

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
					PointCoords* sourceCoords,
					PointColors* sourceColors,
					PointCoords* targetCoords,
					PointColors* targetColors
				  ) 
	{
		m_From = from;
		m_To = to;
		m_NrOfPoints = nrOfPoints;
		m_Metric = metric;
		m_UseRGBData = useRGBData;
		m_WeightRGB = weightRGB;
		m_Indices = indices;
		m_SourceCoords = sourceCoords;
		m_TargetCoords = targetCoords;
		m_SourceColors = sourceColors;
		m_TargetColors = targetColors;
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
	PointCoords* m_SourceCoords;
	PointColors* m_SourceColors;
	PointCoords* m_TargetCoords;
	PointColors* m_TargetColors;
};


#endif // ClosestPointFinderBruteForceCPU_H__
