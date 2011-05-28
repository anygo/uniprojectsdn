#include "ClosestPointFinderBruteForceCPU.h"

#include <limits>
#include <iostream>
#include <boost/thread.hpp>

int
ClosestPointFinderBruteForceCPU::FindClosestPoint(Point6D point) {

	double minSpaceDist = std::numeric_limits<double>::max();
	double minColorDist = std::numeric_limits<double>::max();
	double minDist = std::numeric_limits<double>::max();
	int idx = -1;

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		double spaceDist = 0.;
		double colorDist = 0.;

		switch (m_Metric)
		{
		case ABSOLUTE_DISTANCE:
			spaceDist = std::abs(point.x - m_Target[i].x) + std::abs(point.y - m_Target[i].y) + std::abs(point.z - m_Target[i].z); break;
		case LOG_ABSOLUTE_DISTANCE:
			spaceDist = std::log(std::abs(point.x - m_Target[i].x) + std::abs(point.y - m_Target[i].y) + std::abs(point.z - m_Target[i].z) + 1.0); break;
		case SQUARED_DISTANCE:
			spaceDist = sqrt((point.x - m_Target[i].x)*(point.x - m_Target[i].x) + (point.y - m_Target[i].y)*(point.y - m_Target[i].y) + (point.z - m_Target[i].z)*(point.z - m_Target[i].z));
		}

		if (m_UseRGBData)
		{
			switch (m_Metric)
			{
			case ABSOLUTE_DISTANCE:
				colorDist = std::abs(point.r - m_Target[i].r) + std::abs(point.g - m_Target[i].g) + std::abs(point.b - m_Target[i].b); break;
			case LOG_ABSOLUTE_DISTANCE:
				colorDist = std::log(std::abs(point.r - m_Target[i].r) + std::abs(point.g - m_Target[i].g) + std::abs(point.b - m_Target[i].b) + 1.0); break;
			case SQUARED_DISTANCE:
				colorDist = sqrt((point.r - m_Target[i].r)*(point.r - m_Target[i].r) + (point.g - m_Target[i].g)*(point.g - m_Target[i].g) + (point.b - m_Target[i].b)*(point.b - m_Target[i].b));
			}

		}
		double dist = spaceDist + m_WeightRGB * colorDist;

		if (dist < minDist)
		{
			minSpaceDist = spaceDist;
			minColorDist = colorDist;
			minDist = dist;

			idx = i;
		}
	}

	return idx;
}

int*
ClosestPointFinderBruteForceCPU::FindClosestPoints(Point6D *source) {

	/*for (int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Indices[i] = FindClosestPoint(source[i]);
	}*/

	// use as many threads as there are hardware threads available on the machine
	int numThreads = std::max(1, static_cast<int>(boost::thread::hardware_concurrency()));

	boost::thread* workers = new boost::thread[numThreads];

	for (int i = 0; i < numThreads; ++i)
	{
		// assign a balanced range of points to each thread
		ClosestPointFinderBruteForceCPUWorker worker(
			i*m_NrOfPoints/numThreads,
			(i+1)*m_NrOfPoints/numThreads,
			m_NrOfPoints,
			m_Metric,
			m_UseRGBData,
			m_WeightRGB,
			m_Indices,
			source,
			m_Target
			);

		// start the thread
		workers[i] = boost::thread(worker);
	}

	for (int i = 0; i < numThreads; ++i)
	{
		// wait for the thread to be finished
		workers[i].join();
	}

	// cleanup
	delete [] workers;

	// return the indices used in ICP
	return m_Indices;
}
//----------------------------------------------------------------------------
void
ClosestPointFinderBruteForceCPUWorker::operator()()
{
	for (int j = 0; j < m_NrOfPoints; ++j)
	{
		double minSpaceDist = std::numeric_limits<double>::max();
		double minColorDist = std::numeric_limits<double>::max();
		double minDist = std::numeric_limits<double>::max();
		int idx = -1;

		Point6D point = m_Source[j];

		for (int i = 0; i < m_NrOfPoints; ++i)
		{
			double spaceDist = 0.;
			double colorDist = 0.;

			switch (m_Metric)
			{
			case ABSOLUTE_DISTANCE:
				spaceDist = std::abs(point.x - m_Target[i].x) + std::abs(point.y - m_Target[i].y) + std::abs(point.z - m_Target[i].z); break;
			case LOG_ABSOLUTE_DISTANCE:
				spaceDist = std::log(std::abs(point.x - m_Target[i].x) + std::abs(point.y - m_Target[i].y) + std::abs(point.z - m_Target[i].z) + 1.0); break;
			case SQUARED_DISTANCE:
				spaceDist = sqrt((point.x - m_Target[i].x)*(point.x - m_Target[i].x) + (point.y - m_Target[i].y)*(point.y - m_Target[i].y) + (point.z - m_Target[i].z)*(point.z - m_Target[i].z));
			}

			if (m_UseRGBData)
			{
				switch (m_Metric)
				{
				case ABSOLUTE_DISTANCE:
					colorDist = std::abs(point.r - m_Target[i].r) + std::abs(point.g - m_Target[i].g) + std::abs(point.b - m_Target[i].b); break;
				case LOG_ABSOLUTE_DISTANCE:
					colorDist = std::log(std::abs(point.r - m_Target[i].r) + std::abs(point.g - m_Target[i].g) + std::abs(point.b - m_Target[i].b) + 1.0); break;
				case SQUARED_DISTANCE:
					colorDist = sqrt((point.r - m_Target[i].r)*(point.r - m_Target[i].r) + (point.g - m_Target[i].g)*(point.g - m_Target[i].g) + (point.b - m_Target[i].b)*(point.b - m_Target[i].b));
				}

			}
			double dist = spaceDist + m_WeightRGB * colorDist;

			if (dist < minDist)
			{
				minSpaceDist = spaceDist;
				minColorDist = colorDist;
				minDist = dist;

				idx = i;
			}
		}

		m_Indices[j] = idx;
	}
}