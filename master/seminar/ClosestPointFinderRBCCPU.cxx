#include "ClosestPointFinderRBCCPU.h"

#include <limits>
#include <iostream>

unsigned short*
ClosestPointFinderRBCCPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		float minDist = FLT_MAX;
		Representative* nearestRepresentative;

		// step 1: search nearest representative
		for (std::vector<Representative>::iterator it = m_Representatives.begin(); it != m_Representatives.end(); ++it)
		{
			float dist = DistanceSourceTarget(sourceCoords[i], sourceColors[i], (*it).index);
			if (dist < minDist)
			{
				minDist = dist;
				nearestRepresentative = &(*it);
			}
		}

		// step 2: search nearest neighbor in list of representative
		minDist = FLT_MAX;
		int nearestNeigborIndex = 0;
		for (std::list<unsigned short>::const_iterator it = nearestRepresentative->points.begin(); it != nearestRepresentative->points.end(); ++it)
		{
			float dist = DistanceSourceTarget(sourceCoords[i], sourceColors[i], (*it));
			if (dist < minDist)
			{
				minDist = dist;
				nearestNeigborIndex = (*it);
			}
		}

		m_Indices[i] = nearestNeigborIndex;
		m_Distances[i] = minDist;
	}

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void
ClosestPointFinderRBCCPU::initRBC()
{
	int nrOfReps = std::min(MAX_REPRESENTATIVES, static_cast<int>(sqrt(static_cast<double>(m_NrOfPoints))));

	for (int i = 0; i < nrOfReps; ++i)
	{
		int rep = rand() % m_NrOfPoints;

		// exclude duplicates
		bool duplicate = false;
		for (std::vector<Representative>::const_iterator it = m_Representatives.begin(); it != m_Representatives.end(); ++it)
			if ((*it).index == rep)
				duplicate = true;
		if (duplicate)
		{
			--i;
			continue;
		}

		Representative r;
		r.index = rep;
		m_Representatives.push_back(r);
	}

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		float minDist = FLT_MAX;
		Representative* rep;
		for(std::vector<Representative>::iterator it = m_Representatives.begin(); it != m_Representatives.end(); ++it) {

			float dist = DistanceTargetTarget((*it).index, i);
			
			if (dist < minDist)
			{
				minDist = dist;
				rep = &(*it);
			}
		}
		rep->points.push_back(i);
	}

	DBG << "Random Ball Cover initialized (" << nrOfReps << " Representatives)." << std::endl;
}

float
ClosestPointFinderRBCCPU::DistanceTargetTarget(unsigned short i, unsigned short j)
{
	double x_dist = m_TargetCoords[i].x - m_TargetCoords[j].x; 
	double y_dist = m_TargetCoords[i].y - m_TargetCoords[j].y;
	double z_dist = m_TargetCoords[i].z - m_TargetCoords[j].z;
	double spaceDist; 

	switch (m_Metric)
	{
	case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
	case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
	case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
	}

	if (m_UseRGBData)
	{
		// always use euclidean distance for colors...
		double r_dist = m_TargetColors[i].r - m_TargetColors[j].r; 
		double g_dist = m_TargetColors[i].g - m_TargetColors[j].g;
		double b_dist = m_TargetColors[i].b - m_TargetColors[j].b;
		double colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
		double dist = (1 - m_WeightRGB) * spaceDist + m_WeightRGB * colorDist;
		
		return static_cast<float>(dist);
	} else
	{
		return static_cast<float>(spaceDist);
	}
}

float
ClosestPointFinderRBCCPU::DistanceSourceTarget(PointCoords sourceCoords, PointColors sourceColors, unsigned short targetIndex)
{
	double x_dist = sourceCoords.x - m_TargetCoords[targetIndex].x; 
	double y_dist = sourceCoords.y - m_TargetCoords[targetIndex].y;
	double z_dist = sourceCoords.z - m_TargetCoords[targetIndex].z;
	double spaceDist;

	switch (m_Metric)
	{
	case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
	case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
	case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
	}

	if (m_UseRGBData)
	{
		// always use euclidean distance for colors...
		double r_dist = sourceColors.r - m_TargetColors[targetIndex].r; 
		double g_dist = sourceColors.g - m_TargetColors[targetIndex].g;
		double b_dist = sourceColors.b - m_TargetColors[targetIndex].b;
		double colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
		double dist = (1 - m_WeightRGB) * spaceDist + m_WeightRGB * colorDist;

		return static_cast<float>(dist);
	} else
	{
		return static_cast<float>(spaceDist);
	}
}