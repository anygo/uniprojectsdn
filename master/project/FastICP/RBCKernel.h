#ifndef RBCKERNEL_H__
#define RBCKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "float.h"
#include "RepGPU.h"
#include "defs.h"


// Choose a 1-D distance function that will be applied to all distance computatiosn
// - L2 norm (squared)
#define dist1D(x1, x2) ( (x1-x2)*(x1-x2) )
// - L1 norm
//#define dist1D(x1, x2) ( (x1-x2) >= 0 ? (x1-x2) : -(x1-x2) )


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
__global__ void kernelBuildRBC(float* data, float* weights, RepGPU* reps, unsigned long* NNLists, unsigned long* pointToRep, unsigned long numReps)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float RepBuf[CUDA_BUFFER_SIZE][Dim];
	__shared__ float WeightsBuf[Dim];

	// Pull weights in parallel
	if (threadIdx.x < Dim)
	{
		WeightsBuf[threadIdx.x] = weights[threadIdx.x];
	}

	__syncthreads();


	// Step 1: Search nearest representative
	float minDist = FLT_MAX;
	unsigned long nearestRep = 0;
	for (int rep = 0; rep < numReps; ++rep)
	{
		// Pull representative data in parallel
		if (rep % CUDA_BUFFER_SIZE == 0)
		{
			__syncthreads();

			if ( (threadIdx.x < CUDA_BUFFER_SIZE) && (rep + threadIdx.x < numReps) )
			{
				// Every thread writes data of one representative
				unsigned long repIdx = reps[rep+threadIdx.x].repIdx*Dim;
				for (int d = 0; d < Dim; ++d)
					RepBuf[threadIdx.x][d] = data[repIdx+d];
			}

			__syncthreads();
		}

		float dist = 0;
		for (int d = 0; d < Dim; ++d)
		{
			// Compute weighted distance between tid'th data point and rep'th representative for d'th dimension
			float tmp = dist1D(data[tid*Dim+d], RepBuf[rep % CUDA_BUFFER_SIZE][d]); // dist1D is a macro, e.g. squared L2 norm
			float w = WeightsBuf[d];
			dist += w*tmp;
		}

		if (dist < minDist)
		{
			minDist = dist;
			nearestRep = rep;
		}
	}

	// Write nearest representative to array
	pointToRep[tid] = nearestRep;

	// We only need numReps threads from now on
	if (tid >= numReps) return;

	__syncthreads();


	// Step 2: Count the number of elements for the NN lists
	__shared__ unsigned long counts[MAX_REPS];

	unsigned long cnt = 0;
	for (int i = 0; i < NumPts; ++i)
	{
		if (pointToRep[i] == tid)
			++cnt;
	}
	reps[tid].numPts = cnt;

	// Store to shared memory, s.t. the other threads can read it
	counts[tid] = cnt;

	__syncthreads();


	// Step 3: Compute offset for the tid'th representative's NN list
	unsigned long offset = 0;
	for (int i = 0; i < tid; ++i)
	{
		offset += counts[i];
	}

	// Write modified NNList address to tid'th representative's NNList pointer
	reps[tid].NNList = NNLists+offset;

	__syncthreads();


	// Step 4: Copy point indices to their corresponding NN lists
	int curEl = 0;
	unsigned long* NNList = reps[tid].NNList;
	for (int i = 0; i < NumPts; ++i)
	{
		if (pointToRep[i] == tid)
		{
			NNList[curEl] = i;
			++curEl;
		}
	}
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
__global__ void kernelQueryRBC(float* data, float* weights, float* query, RepGPU* reps, unsigned long* NNIndices, unsigned long numReps)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float RepBuf[CUDA_BUFFER_SIZE][Dim];
	__shared__ float WeightsBuf[Dim];

	// Pull weights in parallel
	if (threadIdx.x < Dim)
	{
		WeightsBuf[threadIdx.x] = weights[threadIdx.x];
	}

	__syncthreads();


	// Step 1: Search nearest representative
	float minDist = FLT_MAX;
	unsigned long nearestRep = 0;
	for (int rep = 0; rep < numReps; ++rep)
	{
		// Pull representative data in parallel
		if (rep % CUDA_BUFFER_SIZE == 0)
		{
			__syncthreads();

			if ( (threadIdx.x < CUDA_BUFFER_SIZE) && (rep + threadIdx.x < numReps) )
			{
				// Every thread fills one representative
				unsigned long repIdx = reps[rep+threadIdx.x].repIdx*Dim;
				for (int d = 0; d < Dim; ++d)
					RepBuf[threadIdx.x][d] = data[repIdx+d];
			}

			__syncthreads();
		}

		float dist = 0;
		for (int d = 0; d < Dim; ++d)
		{
			// Compute weighted distance between tid'th data point and rep'th representative for d'th dimension
			float tmp = dist1D(query[tid*Dim+d], RepBuf[rep % CUDA_BUFFER_SIZE][d]); // dist1D is a macro, e.g. squared L2 norm
			float w = WeightsBuf[d];
			dist += w*tmp;
		}

		if (dist < minDist)
		{
			minDist = dist;
			nearestRep = rep;
		}
	}

	// Step 2: Search nearest point in NN list of nearest representative
	minDist = FLT_MAX;
	unsigned long nn = 0;
	unsigned long numPtsInNNList = reps[nearestRep].numPts;
	unsigned long* NNList = reps[nearestRep].NNList;
	for (int pt = 0; pt < numPtsInNNList; ++pt) 
	{
		float dist = 0;
		unsigned long ptIdx = NNList[pt]*Dim;
		for (int d = 0; d < Dim; ++d)
		{
			// Compute weighted distance
			float tmp = dist1D(query[tid*Dim+d], data[ptIdx+d]); // dist1D is a macro, e.g. squared L2 norm
			float w = WeightsBuf[d];
			dist += w*tmp;
		}

		if (dist < minDist)
		{
			minDist = dist;
			nn = NNList[pt];
		}
	}

	NNIndices[tid] = nn;
}


#endif // RBCKERNEL_H__