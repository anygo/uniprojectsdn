#include "StitchingPluginKernel.h"
#include "defs.h"


// we have to copy the source points only once, because they will be
// transformed directly on the gpu! unfortunately, we do not yet have
// the source points, hence we use that boolean to determine whether
// the data is already on the gpu (after 1st iteration)
bool sourceCopied;





///////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////
	
extern "C"
void TransformPointsDirectlyOnGPU(int nrOfPoints, double transformationMatrix[4][4], PointCoords* writeTo, float* distances)
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
	kernelTransformPointsAndComputeDistance<<<nrOfPoints,1>>>(dev_sourceCoords, dev_distances);
	CUT_CHECK_ERROR("Kernel execution failed (while transforming points)");
	
	// copy distance array to host
	CUDA_SAFE_CALL(cudaMemcpy(distances, dev_distances, nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
	
	// copy transformed points to host
	CUDA_SAFE_CALL(cudaMemcpy(writeTo, dev_sourceCoords, nrOfPoints*sizeof(PointCoords), cudaMemcpyDeviceToHost));
}





///////////////////////////////////////////////////////////////////////////////
// Brute Force
///////////////////////////////////////////////////////////////////////////////

extern "C"
void initGPUBruteForce(PointCoords* targetCoords, PointColors* targetColors, int nrOfPoints) 
{
	sourceCopied = false;

	// allocate memory on gpu
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_indices, nrOfPoints*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_sourceCoords, nrOfPoints*sizeof(PointCoords)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_targetCoords, nrOfPoints*sizeof(PointCoords)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_sourceColors, nrOfPoints*sizeof(PointColors)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_targetColors, nrOfPoints*sizeof(PointColors)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_distances, nrOfPoints*sizeof(float)));
	
	CUDA_SAFE_CALL(cudaMemcpy(dev_targetCoords, targetCoords, nrOfPoints*sizeof(PointCoords), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_targetColors, targetColors, nrOfPoints*sizeof(PointColors), cudaMemcpyHostToDevice));
}

extern "C"
void cleanupGPUBruteForce() 
{
	// free memory
	CUDA_SAFE_CALL(cudaFree(dev_indices));
	CUDA_SAFE_CALL(cudaFree(dev_sourceCoords));
	CUDA_SAFE_CALL(cudaFree(dev_targetCoords));
	CUDA_SAFE_CALL(cudaFree(dev_sourceColors));
	CUDA_SAFE_CALL(cudaFree(dev_targetColors));
	
	CUDA_SAFE_CALL(cudaFree(dev_distances));
}

extern "C"
void FindClosestPointsGPUBruteForce(int nrOfPoints, int metric, bool useRGBData, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, float* distances)
{
	// copy data from host to gpu only if it is not yet copied
	// copy only once, because the data is transformed directly on the gpu!
	if (!sourceCopied)
		CUDA_SAFE_CALL(cudaMemcpy(dev_sourceCoords, sourceCoords, nrOfPoints*sizeof(PointCoords), cudaMemcpyHostToDevice));	
		CUDA_SAFE_CALL(cudaMemcpy(dev_sourceColors, sourceColors, nrOfPoints*sizeof(PointColors), cudaMemcpyHostToDevice));	
	sourceCopied = true;

	// find the closest point for each pixel
	if (useRGBData)
		kernelWithRGBBruteForce<<<nrOfPoints,1>>>(nrOfPoints, metric, weightRGB, dev_indices, dev_sourceCoords, dev_sourceColors, dev_targetCoords, dev_targetColors, dev_distances);
	else
		kernelWithoutRGBBruteForce<<<nrOfPoints,1>>>(nrOfPoints, metric, dev_indices, dev_sourceCoords, dev_targetCoords, dev_distances);
		
	CUT_CHECK_ERROR("Kernel execution failed (while trying to find closest points)");
			
	// copy data from gpu to host
	CUDA_SAFE_CALL(cudaMemcpy(indices, dev_indices, nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(distances, dev_distances, nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
}





///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover
///////////////////////////////////////////////////////////////////////////////

extern "C"
void initGPURBC(PointCoords* targetCoords, PointColors* targetColors, unsigned short* representatives, unsigned short* pointToRep, int nrOfPoints, int nrOfReps)
{
	// same as with BruteForce method
	initGPUBruteForce(targetCoords, targetColors, nrOfPoints);
	
	// plus RBC-specific stuff
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_representatives, nrOfReps*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_pointToRep, nrOfPoints*sizeof(float)));
	
	CUDA_SAFE_CALL(cudaMemcpy(dev_representatives, representatives, nrOfReps*sizeof(unsigned short), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_pointToRep, pointToRep, nrOfPoints*sizeof(unsigned short), cudaMemcpyHostToDevice));
}

extern "C"
void cleanupGPURBC() 
{
	// same as with BruteForce method
	cleanupGPUBruteForce();
	
	// plus RBC-specific cleanup
	CUDA_SAFE_CALL(cudaFree(dev_representatives));
	CUDA_SAFE_CALL(cudaFree(dev_pointToRep));
}

extern "C"
void FindClosestPointsRBC(int nrOfPoints, int nrOfReps, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, float* distances, unsigned short* representatives, unsigned short* pointToRep)
{
	// copy data from host to gpu only if it is not yet copied
	// copy only once, because the data is transformed directly on the gpu!
	if (!sourceCopied)
	{
		CUDA_SAFE_CALL(cudaMemcpy(dev_sourceCoords, sourceCoords, nrOfPoints*sizeof(PointCoords), cudaMemcpyHostToDevice));	
		CUDA_SAFE_CALL(cudaMemcpy(dev_sourceColors, sourceColors, nrOfPoints*sizeof(PointColors), cudaMemcpyHostToDevice));	
	}
	sourceCopied = true;

	// find the closest point for each pixel
	kernelRBC<<<nrOfPoints,1>>>(nrOfPoints, nrOfReps, metric, weightRGB, dev_indices, dev_sourceCoords, dev_sourceColors, dev_targetCoords, dev_targetColors, dev_distances, dev_representatives, dev_pointToRep);	
	//kernelWithRGBBruteForce<<<nrOfPoints,1>>>(nrOfPoints, metric, weightRGB, dev_indices, dev_sourceCoords, dev_sourceColors, dev_targetCoords, dev_targetColors, dev_distances);


	CUT_CHECK_ERROR("Kernel execution failed (while trying to find closest points)");
			
	// copy data from gpu to host
	CUDA_SAFE_CALL(cudaMemcpy(indices, dev_indices, nrOfPoints*sizeof(unsigned short), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(distances, dev_distances, nrOfPoints*sizeof(float), cudaMemcpyDeviceToHost));
	
}