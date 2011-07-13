#include "ClosestPointFinderRBCGPU.h"

#include <iostream>
#include <QTime>

// stupid MS C++ warning - we know what we are doing
#pragma warning( disable : 4996 )


extern "C"
void initGPUMemory(int nrOfPoints, int nrOfReps, float weightRGB, int metric);

extern "C"
void transferToGPU(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors);

extern "C"
void transferRBCData(int nrOfReps, RepGPU* repsGPU, unsigned short* repsIndices);

extern "C"
void PointsToReps(int nrOfReps, unsigned short* pointToRep, unsigned short* reps);

extern "C"
void FindClosestPointsRBC(int nrOfReps, unsigned short* indices, float* distances);

extern "C"
void cleanupGPUCommon();


ClosestPointFinderRBCGPU::ClosestPointFinderRBCGPU(int NrOfPoints, float nrOfRepsFactor) : ClosestPointFinder(NrOfPoints), m_NrOfRepsFactor(nrOfRepsFactor) 
{
	//m_NrOfReps = std::min( MAX_REPRESENTATIVES, static_cast<int>(m_NrOfRepsFactor * sqrt(static_cast<float>(m_NrOfPoints))) );

	// CAUTION!!!
	m_NrOfReps = m_NrOfRepsFactor;
	// CAUTION!!!

	// initialize GPU RBC struct and other data structures
	m_Reps = new unsigned short[m_NrOfReps];
	m_RepsGPU = new RepGPU[m_NrOfReps];	
	m_PointToRep = new unsigned short[m_NrOfPoints];
	m_RepsIndices = new unsigned short[m_NrOfPoints];

	m_Initialized = false;
}

ClosestPointFinderRBCGPU::~ClosestPointFinderRBCGPU()
{ 
	delete[] m_RepsGPU;
	delete[] m_PointToRep;
	delete[] m_Reps;
	delete[] m_RepsIndices;

	// Delete GPU Device Memory
	cleanupGPUCommon();
}

void ClosestPointFinderRBCGPU::SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors) 
{
	if (!m_Initialized)
	{
		initGPUMemory(m_NrOfPoints, m_NrOfReps, m_WeightRGB, m_Metric);
		m_Initialized = true;
	}
	ClosestPointFinder::SetTarget(targetCoords, targetColors, sourceCoords, sourceColors);
	transferToGPU(m_TargetCoords, m_TargetColors, m_SourceCoords, m_SourceColors); // <- we actually dont need our own data struct pointer...
	initRBC();
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
	m_Representatives.clear();

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

		m_Reps[i] = rep;
	}

	// find closest representative to each point on gpu
	PointsToReps(m_NrOfReps, m_PointToRep, m_Reps);

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Representatives[m_PointToRep[i]].points.push_back(i);
	}

	unsigned short* offsetPtr = m_RepsIndices;

	for (int i = 0; i < m_NrOfReps; ++i)
	{
		m_RepsGPU[i].coords = m_TargetCoords[m_Representatives[i].index];
		m_RepsGPU[i].colors = m_TargetColors[m_Representatives[i].index];
		m_RepsGPU[i].nrOfPoints = m_Representatives[i].points.size();

		std::copy(m_Representatives[i].points.begin(), m_Representatives[i].points.end(), offsetPtr);
		offsetPtr += m_RepsGPU[i].nrOfPoints;
	}

	transferRBCData(m_NrOfReps, m_RepsGPU, m_RepsIndices);

}