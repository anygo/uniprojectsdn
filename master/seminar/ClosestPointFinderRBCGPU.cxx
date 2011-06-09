#include "ClosestPointFinderRBCGPU.h"

#include <limits>
#include <iostream>


extern "C"
void initGPURBC(PointCoords* targetCoords, PointColors* targetColors, unsigned short* representatives, unsigned short* pointToRep, int nrOfPoints, int nrOfReps);

extern "C"
void FindClosestPointsRBC(int nrOfPoints, int metric, bool useRGBData, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, float* distances, unsigned short* representatives, unsigned short* pointToRep);



unsigned short*
ClosestPointFinderRBCGPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	FindClosestPointsRBC(m_NrOfPoints, m_Metric, m_UseRGBData, m_WeightRGB, m_Indices, sourceCoords, sourceColors, m_Distances, m_Representatives, m_PointToRep);
	std::cout << "Nach Cuda" << std::endl;
	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------
void
ClosestPointFinderRBCGPU::initRBC()
{
	int nrOfReps = static_cast<int>(sqrt(static_cast<double>(m_NrOfPoints)));
	m_Representatives = new unsigned short[nrOfReps];

	for (int i = 0; i < nrOfReps; ++i)
	{
		int rep = rand() % m_NrOfPoints;

		// exclude duplicates
		bool duplicate = false;
		for (int j = 0; j < i; ++j)
			if (m_Representatives[j] == rep)
				duplicate = true;

		if (duplicate)
		{
			--i;
			continue;
		}

		m_Representatives[i] = rep;
	}

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		float minDist = FLT_MAX;
		unsigned short best;
		for(int j = 0; j < nrOfReps; ++j) {

			float dist = DistanceTargetTarget(m_Representatives[j], i);
			if (dist < minDist)
			{
				minDist = dist;
				best = j;
			}
		}
		m_PointToRep[i] = best;
	}

	std::cout << "Random Ball Cover initialized (" << nrOfReps << " Representatives)." << std::endl;

	initGPURBC(m_TargetCoords, m_TargetColors, m_Representatives, m_PointToRep, m_NrOfPoints, nrOfReps);

	std::cout << "GPU initialized." << std::endl;
}

float
ClosestPointFinderRBCGPU::DistanceTargetTarget(unsigned short i, unsigned short j)
{
	double x_dist = m_TargetCoords[i].x - m_TargetCoords[j].x; 
	double y_dist = m_TargetCoords[i].y - m_TargetCoords[j].y;
	double z_dist = m_TargetCoords[i].z - m_TargetCoords[j].z;
	double spaceDist; 

	switch (m_Metric)
	{
	case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
	case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.0); break;
	case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
	}

	// always use euclidean distance for colors...
	double r_dist = m_TargetColors[i].r - m_TargetColors[j].r; 
	double g_dist = m_TargetColors[i].g - m_TargetColors[j].g;
	double b_dist = m_TargetColors[i].b - m_TargetColors[j].b;
	double colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	double dist = (1 - m_WeightRGB) * spaceDist + m_WeightRGB * colorDist;

	return static_cast<float>(dist);
}
