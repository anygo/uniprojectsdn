#include "ClosestPointFinderBruteForceGPUKernel.h"
#include "defs.h"

#include <stdio.h>
#include <cutil.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>


// global pointers for gpu... 
unsigned short* dev_indices;
Point6D* dev_source;
Point6D* dev_target;

float* dev_distances;
float* dev_transformationMatrix;

// we have to copy the source points only once, because they will be
// transformed directly on the gpu! unfortunately, we do not yet have
// the source points, hence we use that boolean to determine whether
// the data is already on the gpu (after 1st iteration)
bool sourceCopied;
	

extern "C"
void initGPU(Point6D* target, int nrOfPoints) 
{
	sourceCopied = false;

	// allocate memory on gpu
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_indices, nrOfPoints*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_source, nrOfPoints*sizeof(Point6D)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_target, nrOfPoints*sizeof(Point6D)));
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_distances, nrOfPoints*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_transformationMatrix, 16*sizeof(float)));
	
	CUDA_SAFE_CALL(cudaMemcpy(dev_target, target, nrOfPoints*sizeof(Point6D), cudaMemcpyHostToDevice));
}

extern "C"
void cleanupGPU() 
{
	// free memory
	CUDA_SAFE_CALL(cudaFree(dev_indices));
	CUDA_SAFE_CALL(cudaFree(dev_source));
	CUDA_SAFE_CALL(cudaFree(dev_target));
	
	CUDA_SAFE_CALL(cudaFree(dev_distances));
	CUDA_SAFE_CALL(cudaFree(dev_transformationMatrix));
}

extern "C"
void FindClosestPointsCUDA(int nrOfPoints, int metric, bool useRGBData, double weightRGB, unsigned short* indices, Point6D* source)
{
	// copy data from host to gpu only if it is not yet copied
	// copy only once, because the data is transformed directly on the gpu!
	if (!sourceCopied)
		CUDA_SAFE_CALL(cudaMemcpy(dev_source, source, nrOfPoints*sizeof(Point6D), cudaMemcpyHostToDevice));	
	sourceCopied = true;

	// execution
	if (useRGBData)
		kernelWithRGB<<<nrOfPoints,1>>>(nrOfPoints, metric, (float)weightRGB, dev_indices, dev_source, dev_target);
	else
		kernelWithoutRGB<<<nrOfPoints,1>>>(nrOfPoints, metric, dev_indices, dev_source, dev_target);
		
	CUT_CHECK_ERROR("Kernel execution failed (while trying to find closest points)");
			
	// copy data from gpu to host
	CUDA_SAFE_CALL(cudaMemcpy(indices, dev_indices, nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost));
}

extern "C"
void TransformPointsDirectlyOnGPU(int nrOfPoints, double transformationMatrix[4][4], Point6D* writeTo, float* distances)
{
	// allocate memory for transformation matrix (will be stored linearly) and copy it
	float tmp[16];
	tmp[0] = (float)transformationMatrix[0][0];
	tmp[1] = (float)transformationMatrix[0][1];
	tmp[2] = (float)transformationMatrix[0][2];
	tmp[3] = (float)transformationMatrix[0][3];
	tmp[4] = (float)transformationMatrix[1][0];
	tmp[5] = (float)transformationMatrix[1][1];
	tmp[6] = (float)transformationMatrix[1][2];
	tmp[7] = (float)transformationMatrix[1][3];
	tmp[8] = (float)transformationMatrix[2][0];
	tmp[9] = (float)transformationMatrix[2][1];
	tmp[10] = (float)transformationMatrix[2][2];
	tmp[11] = (float)transformationMatrix[2][3];
	tmp[12] = (float)transformationMatrix[3][0];
	tmp[13] = (float)transformationMatrix[3][1];
	tmp[14] = (float)transformationMatrix[3][2];
	tmp[15] = (float)transformationMatrix[3][3];
	
	CUDA_SAFE_CALL(cudaMemcpy(dev_transformationMatrix, tmp, 16*sizeof(float), cudaMemcpyHostToDevice));
	
	// compute transformations
	kernelTransformPoints<<<nrOfPoints,1>>>(dev_source, dev_transformationMatrix, dev_distances);
	CUT_CHECK_ERROR("Kernel execution failed (while transforming points)");
	
	// copy distance array to host
	CUDA_SAFE_CALL(cudaMemcpy(distances, dev_distances, nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
	
	// copy transformed points to host
	CUDA_SAFE_CALL(cudaMemcpy(writeTo, dev_source, nrOfPoints*sizeof(Point6D), cudaMemcpyDeviceToHost));
}
