#include "StitchingPluginKernel.h"
#include "defs.h"


///////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////
extern "C"
void TransformPointsDirectlyOnGPU(double transformationMatrix[4][4], PointCoords* writeTo, float* distances)
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
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_transformationMatrix, tmp, 16*sizeof(float), 0));
	
	// compute transformations
	kernelTransformPointsAndComputeDistance<<<DivUp(host_conf->nrOfPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>();
	CUT_CHECK_ERROR("Kernel execution failed (while transforming points)");
	
	// copy distance array to host
	CUDA_SAFE_CALL(cudaMemcpy(distances, host_conf->distances, host_conf->nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
	
	// copy transformed points to host
	CUDA_SAFE_CALL(cudaMemcpy(writeTo, host_conf->sourceCoords, host_conf->nrOfPoints*sizeof(PointCoords), cudaMemcpyDeviceToHost));
}

extern "C"
void initGPUCommon(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors, float weightRGB, int metric, int nrOfPoints)
{	
	// set up config struct
	host_conf->weightRGB = weightRGB;
	host_conf->metric = metric;
	host_conf->nrOfPoints = nrOfPoints;
	
	// allocate memory on gpu
	CUDA_SAFE_CALL(cudaMalloc((void**)&(host_conf->indices), host_conf->nrOfPoints*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(host_conf->sourceCoords), host_conf->nrOfPoints*sizeof(PointCoords)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(host_conf->targetCoords), host_conf->nrOfPoints*sizeof(PointCoords)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(host_conf->sourceColors), host_conf->nrOfPoints*sizeof(PointColors)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(host_conf->targetColors), host_conf->nrOfPoints*sizeof(PointColors)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(host_conf->distances), host_conf->nrOfPoints*sizeof(float)));
	
	// copy the config struct to gpu
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_conf, host_conf, sizeof(GPUConfig), 0));
	
	// copy actual point cloud data to gpu
	CUDA_SAFE_CALL(cudaMemcpy(host_conf->targetCoords, targetCoords, host_conf->nrOfPoints*sizeof(PointCoords), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(host_conf->targetColors, targetColors, host_conf->nrOfPoints*sizeof(PointColors), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(host_conf->sourceColors, sourceColors, host_conf->nrOfPoints*sizeof(PointColors), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(host_conf->sourceCoords, sourceCoords, host_conf->nrOfPoints*sizeof(PointCoords), cudaMemcpyHostToDevice));
}

extern "C"
void cleanupGPUCommon() 
{
	// free memory
	CUDA_SAFE_CALL(cudaFree(host_conf->indices));
	CUDA_SAFE_CALL(cudaFree(host_conf->distances));
	CUDA_SAFE_CALL(cudaFree(host_conf->sourceCoords));
	CUDA_SAFE_CALL(cudaFree(host_conf->targetCoords));
	CUDA_SAFE_CALL(cudaFree(host_conf->sourceColors));
	CUDA_SAFE_CALL(cudaFree(host_conf->targetColors));
}


///////////////////////////////////////////////////////////////////////////////
// Brute Force
///////////////////////////////////////////////////////////////////////////////

extern "C"
void FindClosestPointsGPUBruteForce(unsigned short* indices, float* distances)
{
	// find the closest point for each pixel
	kernelBruteForce<<<DivUp(host_conf->nrOfPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>();
	
	CUT_CHECK_ERROR("Kernel execution failed (while trying to find closest points)");
			
	// copy data from gpu to host
	CUDA_SAFE_CALL(cudaMemcpy(indices, host_conf->indices, host_conf->nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost));
	
	CUDA_SAFE_CALL(cudaMemcpy(distances, host_conf->distances, host_conf->nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
}

///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover
///////////////////////////////////////////////////////////////////////////////

extern "C"
void initGPURBC(int nrOfReps, RepGPU* repsGPU)
{
	unsigned short* dev_points;
	unsigned short* host_points; // = (unsigned short*)malloc(host_conf->nrOfPoints*sizeof(unsigned short));
	
	//CUDA_SAFE_CALL(cudaHostAlloc( (void**) &host_points, host_conf->nrOfPoints*sizeof(unsigned short), cudaHostAllocDefault ));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_points, host_conf->nrOfPoints*sizeof(unsigned short)));
	
	unsigned short* dev_pointsPtr = dev_points;
	//unsigned short* host_pointsPtr = host_points;
	
	// plus RBC-specific stuff
	for(int i = 0; i < nrOfReps; ++i)
	{
		repsGPU[i].dev_points = dev_pointsPtr; 
		dev_pointsPtr += repsGPU[i].nrOfPoints;
		
		//memcpy(host_pointsPtr, repsGPU[i].points, repsGPU[i].nrOfPoints*sizeof(unsigned short));
	//	host_pointsPtr += repsGPU[i].nrOfPoints;
	}
	
	CUDA_SAFE_CALL(cudaMemcpy(dev_points, repsGPU[0].points, host_conf->nrOfPoints*sizeof(unsigned short), cudaMemcpyHostToDevice));
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_repsGPU, repsGPU, nrOfReps*sizeof(RepGPU), 0));
	
	//CUDA_SAFE_CALL(cudaFreeHost(host_points));
}

extern "C"
void cleanupGPURBC(int nrOfReps, RepGPU* repsGPU) 
{	
	CUDA_SAFE_CALL(cudaFree(repsGPU[0].dev_points)); 
}

extern "C"
void FindClosestPointsRBC(int nrOfReps, unsigned short* indices, float* distances)
{
	///// TIME INITS /////
	//float elapsed;
	//cudaEvent_t start, stop;
	//CUDA_SAFE_CALL(cudaEventCreate(&start));
	//CUDA_SAFE_CALL(cudaEventCreate(&stop));
	//CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	///// TIME START /////

	// find the closest point for each pixel
	//kernelRBC<<<host_conf->nrOfPoints,1>>>(nrOfReps);
	//kernelRBC<<<host_conf->nrOfPoints / CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK>>>(nrOfReps);
	kernelRBC<<<DivUp(host_conf->nrOfPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(nrOfReps);
	
	CUT_CHECK_ERROR("Kernel execution failed (while trying to find closest points)");
			
	// copy data from gpu to host
	CUDA_SAFE_CALL(cudaMemcpy(indices, host_conf->indices, host_conf->nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(distances, host_conf->distances, host_conf->nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
	
	///// TIME END /////
	//CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	//CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	//CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, stop));
	//printf("Time for FindClosestPoints RBC: %3.3f ms\n", elapsed);
}

extern "C"
void PointsToReps(int nrOfReps, unsigned short* pointToRep, unsigned short* reps)
{
	unsigned short* dev_pointToRep;
	unsigned short* dev_reps;
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_pointToRep, host_conf->nrOfPoints*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_reps, host_conf->nrOfPoints*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMemcpy(dev_reps, reps, nrOfReps*sizeof(unsigned short), cudaMemcpyHostToDevice));
	
	
	kernelPointsToReps<<<DivUp(host_conf->nrOfPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(nrOfReps, dev_pointToRep, dev_reps);
	
	CUDA_SAFE_CALL(cudaMemcpy(pointToRep, dev_pointToRep, host_conf->nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(dev_pointToRep));
	CUDA_SAFE_CALL(cudaFree(dev_reps));
}