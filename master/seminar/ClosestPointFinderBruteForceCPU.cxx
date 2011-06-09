#include "ClosestPointFinderBruteForceCPU.h"

#include <limits>
#include <iostream>

unsigned short*
ClosestPointFinderBruteForceCPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors) {
	
	// create some worker threads and do the job
	int numThreads = 1;
	if (m_Multithreaded)
	{
		// use as many threads as there are hardware threads available on the machine
		numThreads = std::max(1, QThread::idealThreadCount()); 
	}

	ClosestPointFinderBruteForceCPUWorker* workers = new ClosestPointFinderBruteForceCPUWorker[numThreads];

	for (int i = 0; i < numThreads; ++i)
	{
		// assign a balanced range of points to each thread
		workers[i].setConfig(
			i*m_NrOfPoints/numThreads,
			(i+1)*m_NrOfPoints/numThreads,
			m_NrOfPoints,
			m_Metric,
			m_UseRGBData,
			m_WeightRGB,
			m_Indices,
			sourceCoords,
			sourceColors,
			m_TargetCoords,
			m_TargetColors,
			m_Distances
			);

		// start the thread
		workers[i].start();
	}

	for (int i = 0; i < numThreads; ++i)
	{
		// wait for the thread to be finished
		workers[i].wait();
	}

	// cleanup
	delete [] workers;

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void
ClosestPointFinderBruteForceCPUWorker::run()
{
	for (int j = 0; j < m_NrOfPoints; ++j)
	{
		double minSpaceDist = std::numeric_limits<double>::max();
		double minColorDist = std::numeric_limits<double>::max();
		double minDist = std::numeric_limits<double>::max();
		int idx = -1;

		PointCoords coords = m_SourceCoords[j];
		PointColors colors = m_SourceColors[j];

		for (int i = 0; i < m_NrOfPoints; ++i)
		{
			double spaceDist = 0.;
			double colorDist = 0.;

			switch (m_Metric)
			{
			case ABSOLUTE_DISTANCE:
				spaceDist = std::abs(coords.x - m_TargetCoords[i].x) + std::abs(coords.y - m_TargetCoords[i].y) + std::abs(coords.z - m_TargetCoords[i].z); break;
			case LOG_ABSOLUTE_DISTANCE:
				spaceDist = std::log(std::abs(coords.x - m_TargetCoords[i].x) + std::abs(coords.y - m_TargetCoords[i].y) + std::abs(coords.z - m_TargetCoords[i].z) + 1.f); break;
			case SQUARED_DISTANCE:
				spaceDist = (coords.x - m_TargetCoords[i].x)*(coords.x - m_TargetCoords[i].x) + (coords.y - m_TargetCoords[i].y)*(coords.y - m_TargetCoords[i].y) + (coords.z - m_TargetCoords[i].z)*(coords.z - m_TargetCoords[i].z);
			}

			if (m_UseRGBData)
			{
				switch (m_Metric)
				{
				case ABSOLUTE_DISTANCE:
					colorDist = std::abs(colors.r - m_TargetColors[i].r) + std::abs(colors.g - m_TargetColors[i].g) + std::abs(colors.b - m_TargetColors[i].b); break;
				case LOG_ABSOLUTE_DISTANCE:
					colorDist = std::log(std::abs(colors.r - m_TargetColors[i].r) + std::abs(colors.g - m_TargetColors[i].g) + std::abs(colors.b - m_TargetColors[i].b) + 1.f); break;
				case SQUARED_DISTANCE:
					colorDist = (colors.r - m_TargetColors[i].r)*(colors.r - m_TargetColors[i].r) + (colors.g - m_TargetColors[i].g)*(colors.g - m_TargetColors[i].g) + (colors.b - m_TargetColors[i].b)*(colors.b - m_TargetColors[i].b);
				}

			}
			double dist = (1 - m_WeightRGB) * spaceDist + m_WeightRGB * colorDist;

			if (dist < minDist)
			{
				minSpaceDist = spaceDist;
				minColorDist = colorDist;
				minDist = dist;

				idx = i;
			}
		}

		m_Indices[j] = idx;
		m_Distances[j] = minDist;
	}
}