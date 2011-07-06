#include "FastStitchingPluginKernel.h"
#include "defs.h"



///////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////
extern "C"
void
CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray)
{
	// Set input image texture parameters and bind texture to the array. Texture is defined in the kernel
	InputImageTexture.addressMode[0] = cudaAddressModeClamp;
	InputImageTexture.addressMode[1] = cudaAddressModeClamp;
	InputImageTexture.filterMode = cudaFilterModePoint;
	InputImageTexture.normalized = false;
	cutilSafeCall(cudaBindTextureToArray(InputImageTexture, InputImageArray));
	
	// Kernel Invocation
	dim3 DimBlock(16, 16);
	dim3 DimGrid(DivUp(FRAME_SIZE_X, DimBlock.x), DivUp(FRAME_SIZE_Y, DimBlock.y));
	CUDARangeToWorldKernel<16,16><<<DimGrid,DimBlock>>>(duplicate);

	// Unbind texture
	cutilSafeCall(cudaUnbindTexture(InputImageTexture));

	CUT_CHECK_ERROR("Kernel execution failed");
}

extern "C"
void CUDATransformPoints(double transformationMatrix[4][4], float4* toBeTransformed, int numPoints, float* distances)
{
	//printf("CUDATransformPoints\n");
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
	
	kernelTransformPointsAndComputeDistance<<<DivUp(numPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(toBeTransformed, distances, numPoints);
	CUT_CHECK_ERROR("Kernel execution failed (while transforming points)");
}

extern "C"
void
CUDAExtractLandmarks(int numLandmarks, float4* devWCsIn, unsigned int* devIndicesIn, float4* devLandmarksOut)
{
	//printf("CUDAExtractLandmarks\n");
	kernelExtractLandmarks<<<DivUp(numLandmarks, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(devWCsIn, devIndicesIn, devLandmarksOut);
}

///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover
///////////////////////////////////////////////////////////////////////////////

extern "C"
void initGPURBC(int nrOfReps, RepGPU* repsGPU)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_repsGPU, repsGPU, nrOfReps*sizeof(RepGPU), 0));
}

extern "C"
void CUDAFindClosestPointsRBC(int nrOfPoints, int nrOfReps, unsigned int* indices, float* distances, float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors)
{
	// find the closest point for each pixel
	kernelRBC<<<DivUp(nrOfPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(nrOfReps, indices, distances, targetCoords, targetColors, sourceCoords, sourceColors);
	
	CUT_CHECK_ERROR("Kernel execution failed (while trying to find closest points)");
}

extern "C"
void CUDAPointsToReps(int nrOfPoints, int nrOfReps, float4* devTargetCoords, float4* devTargetColors, unsigned int* devRepIndices, unsigned int* devPointToRep)
{
	kernelPointsToReps<<<DivUp(nrOfPoints, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(nrOfReps, devTargetCoords, devTargetColors, devRepIndices, devPointToRep);
}