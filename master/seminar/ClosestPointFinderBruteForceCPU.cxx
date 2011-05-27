#include "ClosestPointFinderBruteForceCPU.h"

#include <limits>

int ClosestPointFinderBruteForceCPU::FindClosestPoint(Point6D point) {

	double minDist = std::numeric_limits<double>::max();
	int idx = -1;

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		double dist_tmp = 0.;
		switch (m_Metric)
		{
		case ABSOLUTE_DISTANCE:
			dist_tmp = std::abs(point.x - m_Target[i].x) + std::abs(point.y - m_Target[i].y) + std::abs(point.z - m_Target[i].z); break;
		case LOG_ABSOLUTE_DISTANCE:
			dist_tmp = std::log(std::abs(point.x - m_Target[i].x) + std::abs(point.y - m_Target[i].y) + std::abs(point.z - m_Target[i].z) + 1.0); break;
		case SQUARED_DISTANCE:
			dist_tmp = sqrt((point.x - m_Target[i].x)*(point.x - m_Target[i].x) + (point.y - m_Target[i].y)*(point.y - m_Target[i].y) + (point.z - m_Target[i].z)*(point.z - m_Target[i].z));
		}

		if (dist_tmp < minDist)
		{
			minDist = dist_tmp;
			idx = i;
		}
	}

	return idx;
}

int* ClosestPointFinderBruteForceCPU::FindClosestPoints(Point6D *source) {

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Indices[i] = FindClosestPoint(source[i]);
	}

	return m_Indices;
}

