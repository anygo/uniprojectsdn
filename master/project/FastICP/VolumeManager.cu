#include "VolumeManagerKernel.h"
#include "defs.h"
#include "ritkCudaMacros.h"
#include <stdio.h>


//----------------------------------------------------------------------------
extern "C"
void CUDAAddPointsToVolumeVoxelToPoint(float* points, unsigned char* voxels, float* config, unsigned long numPts, int numVoxels)
{
	kernelAddPointsToVolumeVoxelToPoint<<<DIVUP(numVoxels, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, voxels, config, numPts);
}


//----------------------------------------------------------------------------
extern "C"
void CUDAAddPointsToVolumePointToVoxel(float* points, unsigned char* voxels, float* config, unsigned long numPts)
{
	kernelAddPointsToVolumePointToVoxel<<<DIVUP(numPts, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, voxels, config, numPts);
}