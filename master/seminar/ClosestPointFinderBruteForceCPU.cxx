#include "ClosestPointFinderBruteForceCPU.h"

#include <limits>
#include <iostream>

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

	//std::cout << "Space dist: " << minSpaceDist << " \t Color dist: " << minColorDist << std::endl;

	return idx;
}

int*
ClosestPointFinderBruteForceCPU::FindClosestPoints(Point6D *source) {

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Indices[i] = FindClosestPoint(source[i]);
	}

	return m_Indices;
}

