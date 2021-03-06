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
	inline bool usesGPU() { return false; }

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
					float weightRGB,
					unsigned short* indices,
					PointCoords* sourceCoords,
					PointColors* sourceColors,
					PointCoords* targetCoords,
					PointColors* targetColors,
					float* distances
				  ) 
	{
		m_From = from;
		m_To = to;
		m_NrOfPoints = nrOfPoints;
		m_Metric = metric;
		m_WeightRGB = weightRGB;
		m_Indices = indices;
		m_SourceCoords = sourceCoords;
		m_TargetCoords = targetCoords;
		m_SourceColors = sourceColors;
		m_TargetColors = targetColors;
		m_Distances = distances;
	}

protected:
	void run();

	int m_From;
	int m_To;
	int m_NrOfPoints;
	int m_Metric;
	float m_WeightRGB;
	unsigned short* m_Indices;
	PointCoords* m_SourceCoords;
	PointColors* m_SourceColors;
	PointCoords* m_TargetCoords;
	PointColors* m_TargetColors;
	float* m_Distances;
};


#endif // ClosestPointFinderBruteForceCPU_H__
