#include "ClosestPointFinderRBCGPU.h"

#include <iostream>
#include <QTime>

// stupid MS C++ warning - we know what we are doing
#pragma warning( disable : 4996 )


extern "C"
void CUDAPointsToReps(int nrOfPoints, int nrOfReps, float4* devTargetCoords, float4* devTargetColors, unsigned int* devRepIndices, unsigned int* devPointToRep);

extern "C"
void initGPURBC(int nrOfReps, RepGPU* repsGPU);

extern "C"
void CUDAFindClosestPointsRBC(int nrOfPoints, int nrOfReps, unsigned int* indices, float* distances, float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors);
///////////////


extern "C"
void FindClosestPointsRBC(int nrOfReps, unsigned short* indices, float* distances);


ClosestPointFinderRBCGPU::ClosestPointFinderRBCGPU(int NrOfPoints, float weightRGB, float nrOfRepsFactor) 
	: ClosestPointFinder(NrOfPoints, weightRGB), m_NrOfRepsFactor(nrOfRepsFactor) 
{
	//std::cout << "ClosestPointFinderRBCGPU" << std::endl;

	m_NrOfReps = std::min( MAX_REPRESENTATIVES, static_cast<int>(m_NrOfRepsFactor * sqrt(static_cast<float>(m_NrOfPoints))) );

	// initialize GPU RBC struct and other data structures
	m_RepIndices = new unsigned int[m_NrOfReps];
	m_RepsGPU = new RepGPU[m_NrOfReps];	
	m_PointToRep = new unsigned int[m_NrOfPoints];

	// RBC GPU pointer
	cutilSafeCall(cudaMalloc((void**)&(m_devDistances), m_NrOfPoints*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&(m_devIndices), m_NrOfPoints*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devPointToRep), m_NrOfPoints*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devRepIndices), m_NrOfReps*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devReps), m_NrOfReps*sizeof(unsigned int)));

	m_Initialized = true;
}

ClosestPointFinderRBCGPU::~ClosestPointFinderRBCGPU()
{ 
	//std::cout << "~ClosestPointFinderRBCGPU" << std::endl;
	delete[] m_RepsGPU;
	delete[] m_PointToRep;
	delete[] m_RepIndices;

	// Delete GPU Device Memory
	cutilSafeCall(cudaFree(m_devDistances));
	cutilSafeCall(cudaFree(m_devIndices));
	cutilSafeCall(cudaFree(m_devPointToRep));
	cutilSafeCall(cudaFree(m_devRepIndices));
	cutilSafeCall(cudaFree(m_devReps));
}

void
ClosestPointFinderRBCGPU::Initialize(float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors)  
{
	//std::cout << "Initialize" << std::endl;

	ClosestPointFinder::Initialize(targetCoords, targetColors, sourceCoords, sourceColors); 
	InitializeRBC();
}

void
ClosestPointFinderRBCGPU::FindClosestPoints(unsigned int* indices, float* distances)
{
	//std::cout << "FindClosestPoints" << std::endl;

	// returns the indices which will then be used in the icp algorithm
	CUDAFindClosestPointsRBC(m_NrOfPoints, m_NrOfReps, m_devIndices, m_devDistances, m_devTargetCoords, m_devTargetColors, m_devSourceCoords, m_devSourceColors);

	// Copy distances and indices back to cpu for estimating transformation matrix...
	cutilSafeCall(cudaMemcpy(indices, m_devIndices, m_NrOfPoints*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(distances, m_devDistances, m_NrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------
void
ClosestPointFinderRBCGPU::InitializeRBC()
{
	//std::cout << "InitializeRBC" << std::endl;

	// Clear vector content
	m_Representatives.clear();

	// Generate Representant indices randomly in the range from [0 #Landmarks]
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

		m_RepIndices[i] = rep;
	}

	// Copy the indices of the reps to gpu
	cutilSafeCall(cudaMemcpy(m_devRepIndices, m_RepIndices, m_NrOfReps*sizeof(unsigned int), cudaMemcpyHostToDevice));

	// find closest representative to each point on gpu
	CUDAPointsToReps(m_NrOfPoints, m_NrOfReps, m_devTargetCoords, m_devTargetColors, m_devRepIndices, m_devPointToRep);

	// Copy the pointToRep mapping to cpu
	cutilSafeCall(cudaMemcpy(m_PointToRep, m_devPointToRep, m_NrOfPoints*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Representatives[m_PointToRep[i]].points.push_back(i);
	}

	// Create space for the owner lists of the representatives
	unsigned int* dev_points;
	unsigned int* repOwnerList = new unsigned int[m_NrOfPoints];
	unsigned int* offsetPtr = repOwnerList;

	cutilSafeCall(cudaMalloc((void**)&dev_points, m_NrOfPoints*sizeof(unsigned int)));
	unsigned int* dev_pointsPtr = dev_points;

	for (int i = 0; i < m_NrOfReps; ++i)
	{
		m_RepsGPU[i].index = m_Representatives[i].index;
		m_RepsGPU[i].nrOfPoints = m_Representatives[i].points.size();
		m_RepsGPU[i].dev_points = dev_pointsPtr; 

		std::copy(m_Representatives[i].points.begin(), m_Representatives[i].points.end(), offsetPtr);

		dev_pointsPtr += m_RepsGPU[i].nrOfPoints;
		offsetPtr += m_RepsGPU[i].nrOfPoints;
	}

	cutilSafeCall(cudaMemcpy(dev_points, repOwnerList, m_NrOfPoints*sizeof(unsigned int), cudaMemcpyHostToDevice));

	// Just copies the Reps struct onto gpu
	initGPURBC(m_NrOfReps, m_RepsGPU);

	delete[] repOwnerList;

}