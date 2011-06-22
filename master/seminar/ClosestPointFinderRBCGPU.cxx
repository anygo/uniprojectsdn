#include "ClosestPointFinderRBCGPU.h"

#include <limits>
#include <iostream>
#include <algorithm>
#include <QTime>

// stupid MS C++ warning - we know what we are doing
#pragma warning( disable : 4996 )

extern "C"
void initGPUCommon(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors, float weightRGB, int metric, int nrOfPoints);


extern "C"
void initGPURBC(int nrOfReps, RepGPU* repsGPU);

extern "C"
void PointsToReps(int nrOfReps, unsigned short* pointToRep, unsigned short* reps);

extern "C"
void cleanupGPURBC(int nrOfReps, RepGPU* repsGPU);

extern "C"
void FindClosestPointsRBC(int nrOfReps, unsigned short* indices, float* distances);


ClosestPointFinderRBCGPU::~ClosestPointFinderRBCGPU()
{ 
	cleanupGPURBC(m_NrOfReps, m_RepsGPU);

	for(int i = 0; i < m_NrOfReps; ++i) 
	{
		delete[] m_RepsGPU[i].points;
	}	
	delete[] m_RepsGPU; 
}

unsigned short*
ClosestPointFinderRBCGPU::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	FindClosestPointsRBC(m_NrOfReps, m_Indices, m_Distances);

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------

void
ClosestPointFinderRBCGPU::initRBC()
{
	m_NrOfReps = std::min(MAX_REPRESENTATIVES, static_cast<int>(m_NrOfRepsFactor * sqrt(static_cast<float>(m_NrOfPoints))));

	// initialize GPU for RBC initialization
	initGPUCommon(m_TargetCoords, m_TargetColors, m_SourceCoords, m_SourceColors, m_WeightRGB, m_Metric, m_NrOfPoints);

	unsigned short* reps = new unsigned short[m_NrOfReps];
	unsigned short* pointToRep = new unsigned short[m_NrOfPoints];


	for (int i = 0; i < m_NrOfReps; ++i)
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

		reps[i] = rep;
	}

	// find closest representative to each point on gpu
	PointsToReps(m_NrOfReps, pointToRep, reps);

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Representatives[pointToRep[i]].points.push_back(i);
	}

	DBG << "Random Ball Cover initialized (" << m_NrOfReps << " Representatives)." << std::endl;

	// initialize GPU RBC struct
	m_RepsGPU = new RepGPU[m_NrOfReps];
	
	for (int i = 0; i < m_NrOfReps; ++i)
	{
		m_RepsGPU[i].coords = m_TargetCoords[m_Representatives[i].index];
		m_RepsGPU[i].colors = m_TargetColors[m_Representatives[i].index];
		m_RepsGPU[i].nrOfPoints = m_Representatives[i].points.size();
		m_RepsGPU[i].points = new unsigned short[m_RepsGPU[i].nrOfPoints];
		std::copy(m_Representatives[i].points.begin(), m_Representatives[i].points.end(), m_RepsGPU[i].points);
	}

	DBG << "Copying data to gpu..." << std::endl;
	initGPURBC(m_NrOfReps, m_RepsGPU);

	delete[] pointToRep;
	delete[] reps;
}

float
ClosestPointFinderRBCGPU::DistanceTargetTarget(unsigned short i, unsigned short j)
{
	float x_dist = m_TargetCoords[i].x - m_TargetCoords[j].x; 
	float y_dist = m_TargetCoords[i].y - m_TargetCoords[j].y;
	float z_dist = m_TargetCoords[i].z - m_TargetCoords[j].z;
	float spaceDist; 

	switch (m_Metric)
	{
	case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
	case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
	case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
	}


	// always use euclidean distance for colors...
	float r_dist = m_TargetColors[i].r - m_TargetColors[j].r; 
	float g_dist = m_TargetColors[i].g - m_TargetColors[j].g;
	float b_dist = m_TargetColors[i].b - m_TargetColors[j].b;
	float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	float dist = (1 - m_WeightRGB) * spaceDist + m_WeightRGB * colorDist;

	return static_cast<float>(dist);
}