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
int* dev_indices;
Point6D* dev_source;
Point6D* dev_target;
	

extern "C"
void initGPU(Point6D* target, int nrOfPoints) 
{
	// allocate memory on gpu
	cudaMalloc((void**)&dev_indices, nrOfPoints*sizeof(int));
	cudaMalloc((void**)&dev_source, nrOfPoints*sizeof(Point6D));
	cudaMalloc((void**)&dev_target, nrOfPoints*sizeof(Point6D));
	
	cudaMemcpy(dev_target, target, nrOfPoints*sizeof(Point6D), cudaMemcpyHostToDevice);
}

extern "C"
void cleanupGPU() 
{
	// free memory
	cudaFree(dev_indices);
	cudaFree(dev_source);
	cudaFree(dev_target);
}

extern "C"
void FindClosestPointsCUDA(int nrOfPoints, int metric, bool useRGBData, double weightRGB, int* indices, Point6D* source)
{

	// copy data from host to gpu
	cudaMemcpy(dev_source, source, nrOfPoints*sizeof(Point6D), cudaMemcpyHostToDevice);	

	// execution
	if (useRGBData)
		kernelWithRGB<<<nrOfPoints,1>>>(nrOfPoints, metric, (float)weightRGB, dev_indices, dev_source, dev_target);
	else
		kernelWithoutRGB<<<nrOfPoints,1>>>(nrOfPoints, metric, dev_indices, dev_source, dev_target);
			
	// copy data from gpu to host
	cudaMemcpy(indices, dev_indices, nrOfPoints*sizeof(int), cudaMemcpyDeviceToHost);	
	
}