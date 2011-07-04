#include "ClosestPointFinderRBCGPU.h"

#include <iostream>
#include <QTime>

// stupid MS C++ warning - we know what we are doing
#pragma warning( disable : 4996 )


extern "C"
void initGPUMemory(int nrOfPoints, float weightRGB, int metric);

extern "C"
void transferToGPU(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors);

extern "C"
void cleanupGPUCommon();

/////////////////////

extern "C"
void initGPUCommon(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors, float weightRGB, int metric, int nrOfPoints);

extern "C"
void initGPURBC(int nrOfReps, RepGPU* repsGPU, unsigned short* repsIndices);

extern "C"
void cleanupGPURBC(); 

extern "C"
void PointsToReps(int nrOfReps, unsigned short* pointToRep, unsigned short* reps);

extern "C"
void FindClosestPointsRBC(int nrOfReps, unsigned short* indices, float* distances);


ClosestPointFinderRBCGPU::ClosestPointFinderRBCGPU(int NrOfPoints, int metric, int weightRGB, float nrOfRepsFactor) 
	: ClosestPointFinder(NrOfPoints, metric, weightRGB), m_NrOfRepsFactor(nrOfRepsFactor) 
{
	m_NrOfReps = std::min( MAX_REPRESENTATIVES, static_cast<int>(m_NrOfRepsFactor * sqrt(static_cast<float>(m_NrOfPoints))) );

	// initialize GPU RBC struct and other data structures
	m_Reps = new unsigned int[m_NrOfReps];
	m_RepsGPU = new RepGPU[m_NrOfReps];	
	m_PointToRep = new unsigned int[m_NrOfPoints];
	m_RepsIndices = new unsigned int[m_NrOfPoints];

	m_Initialized = false;
}

ClosestPointFinderRBCGPU::~ClosestPointFinderRBCGPU()
{ 
	delete[] m_RepsGPU;
	delete[] m_PointToRep;
	delete[] m_Reps;
	delete[] m_RepsIndices;

	// Delete GPU Device Memory
	cleanupGPURBC();
	cleanupGPUCommon();
}

void ClosestPointFinderRBCGPU::Initialize(float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors)  
{

	ClosestPointFinder::Initialize(targetCoords, targetColors, sourceCoords, sourceColors); 

	if (!m_Initialized)
	{
		initGPUMemory(m_NrOfPoints, m_WeightRGB, m_Metric);
		m_Initialized = true;
	}

	transferToGPU(m_TargetCoords, m_TargetColors, m_SourceCoords, m_SourceColors);
	initRBC();
}

void
ClosestPointFinderRBCGPU::FindClosestPoints(int* indices, float* distances)
{
	// returns the indices which will then be used in the icp algorithm
	FindClosestPointsRBC(m_NrOfReps, indices, distances);	
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

	unsigned int* offsetPtr = m_RepsIndices;

	for (int i = 0; i < m_NrOfReps; ++i)
	{
		m_RepsGPU[i].coords = m_TargetCoords[m_Representatives[i].index];
		m_RepsGPU[i].colors = m_TargetColors[m_Representatives[i].index];
		m_RepsGPU[i].nrOfPoints = m_Representatives[i].points.size();

		std::copy(m_Representatives[i].points.begin(), m_Representatives[i].points.end(), offsetPtr);
		offsetPtr += m_RepsGPU[i].nrOfPoints;
	}

	initGPURBC(m_NrOfReps, m_RepsGPU, m_RepsIndices);

}