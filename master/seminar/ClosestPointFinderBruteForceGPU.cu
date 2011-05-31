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
bool sourceCopied;
	

extern "C"
void initGPU(Point6D* target, int nrOfPoints) 
{
	sourceCopied = false;

	// allocate memory on gpu
	cudaMalloc((void**)&dev_indices, nrOfPoints*sizeof(unsigned short));
	cudaMalloc((void**)&dev_source, nrOfPoints*sizeof(Point6D));
	cudaMalloc((void**)&dev_target, nrOfPoints*sizeof(Point6D));
	
	cudaMalloc((void**)&dev_distances, nrOfPoints*sizeof(float));
	cudaMalloc((void**)&dev_transformationMatrix, 16*sizeof(float));
	
	cudaMemcpy(dev_target, target, nrOfPoints*sizeof(Point6D), cudaMemcpyHostToDevice);
}

extern "C"
void cleanupGPU() 
{
	// free memory
	cudaFree(dev_indices);
	cudaFree(dev_source);
	cudaFree(dev_target);
	
	cudaFree(dev_distances);
	cudaFree(dev_transformationMatrix);
}

extern "C"
void FindClosestPointsCUDA(int nrOfPoints, int metric, bool useRGBData, double weightRGB, unsigned short* indices, Point6D* source)
{
	// copy data from host to gpu only if it is not yet copied
	// copy only once, because the data is transformed directly on the gpu!
	if (!sourceCopied)
		cudaMemcpy(dev_source, source, nrOfPoints*sizeof(Point6D), cudaMemcpyHostToDevice);	
	sourceCopied = true;

	// execution
	if (useRGBData)
		kernelWithRGB<<<nrOfPoints,1>>>(nrOfPoints, metric, (float)weightRGB, dev_indices, dev_source, dev_target);
	else
		kernelWithoutRGB<<<nrOfPoints,1>>>(nrOfPoints, metric, dev_indices, dev_source, dev_target);
			
	// copy data from gpu to host
	cudaMemcpy(indices, dev_indices, nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost);
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
	
	cudaMemcpy(dev_transformationMatrix, tmp, 16*sizeof(float), cudaMemcpyHostToDevice);
	
	// compute transformations
	kernelTransformPoints<<<nrOfPoints,1>>>(dev_source, dev_transformationMatrix, dev_distances);
	
	// copy distance array to host
	cudaMemcpy(distances, dev_distances, nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost);
	
	// copy transformed points to host
	cudaMemcpy(writeTo, dev_source, nrOfPoints*sizeof(Point6D), cudaMemcpyDeviceToHost);
	
}