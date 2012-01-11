#include "KinectDataManagerKernel.h"
#include "ritkCudaMacros.h"
#include "defs.h"


//----------------------------------------------------------------------------
extern "C"
void CUDARangeToWorld(float* pointsOut, const cudaArray* inputImageArray)
{
	// Set input image texture parameters and bind texture to the array. Texture is defined in the kernel
	InputImageTexture.addressMode[0] = cudaAddressModeClamp;
	InputImageTexture.addressMode[1] = cudaAddressModeClamp;
	InputImageTexture.filterMode = cudaFilterModePoint;
	InputImageTexture.normalized = false;
	ritkCudaSafeCall( cudaBindTextureToArray(InputImageTexture, inputImageArray) );
	
	// Kernel Invocation
	dim3 DimBlock(16, 16);
	dim3 DimGrid(DIVUP(KINECT_IMAGE_WIDTH, DimBlock.x), DIVUP(KINECT_IMAGE_HEIGHT, DimBlock.y));
	CUDARangeToWorldKernel<16,16><<<DimGrid,DimBlock>>>(pointsOut);

	// Unbind texture
	ritkCudaSafeCall( cudaUnbindTexture(InputImageTexture) );
}


//----------------------------------------------------------------------------
extern "C"
void CUDAExtractLandmarks(float* landmarksOut, float* pointsIn, unsigned long* indices, unsigned long numLandmarks)
{
	kernelExtractLandmarks<<<DIVUP(numLandmarks, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(landmarksOut, pointsIn, indices, numLandmarks);
}