#include "VolumeManagerKernel.h"
#include "defs.h"
#include "ritkCudaMacros.h"
#include <stdio.h>


//----------------------------------------------------------------------------
extern "C"
void CUDAAddPointsToVolume(float* points, float* voxels, float* origin, unsigned long numPts, unsigned int dimSize, unsigned int spacing)
{
	kernelAddPointsToVolume<<<DIVUP(dimSize*dimSize*dimSize, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, voxels, origin, numPts, dimSize, spacing);
}