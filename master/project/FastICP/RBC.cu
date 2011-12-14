#include "RBCKernel.h"
#include "defs.h"
#include <stdio.h>


//----------------------------------------------------------------------------
extern "C"
void CUDABuildRBC(float* data, float* weights, RepGPU* reps, unsigned long* NNLists, unsigned long* pointToRep, unsigned long numReps, unsigned long numPts, unsigned long dim)
{
	if (dim == 6)
	{
		if (numPts == 32)
			kernelBuildRBC<32,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 128)
			kernelBuildRBC<128,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 256)
			kernelBuildRBC<256,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 512)
			kernelBuildRBC<512,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 1024)
			kernelBuildRBC<1024,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 2048)
			kernelBuildRBC<2048,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 4096)
			kernelBuildRBC<4096,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 8192)
			kernelBuildRBC<8192,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else if (numPts == 16384)
			kernelBuildRBC<16384,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, reps, NNLists, pointToRep, numReps);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else {
		printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
}


//----------------------------------------------------------------------------
extern "C"
void CUDAQueryRBC(float* data, float* weights, float* query, RepGPU* reps, unsigned long* NNIndices, unsigned long numReps, unsigned long numPts, unsigned long dim)
{
	if (dim == 6)
	{
		if (numPts == 32)
			kernelQueryRBC<32,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 128)
			kernelQueryRBC<128,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 256)
			kernelQueryRBC<256,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 512)
			kernelQueryRBC<512,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 1024)
			kernelQueryRBC<1024,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 2048)
			kernelQueryRBC<2048,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 4096)
			kernelQueryRBC<4096,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 8192)
			kernelQueryRBC<8192,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else if (numPts == 16384)
			kernelQueryRBC<16384,6><<<DivUp(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(data, weights, query, reps, NNIndices, numReps);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else {
		printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
}