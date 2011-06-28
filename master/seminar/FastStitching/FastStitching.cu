#include "FastStitchingKernel.h"
#include <cutil.h>

#define CUDA_THREADS_PER_BLOCK 128

extern "C"
void
CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray, int w, int h)
{
	// Set input image texture parameters and bind texture to the array. Texture is defined in the kernel
	InputImageTexture.addressMode[0] = cudaAddressModeClamp;
	InputImageTexture.addressMode[1] = cudaAddressModeClamp;
	InputImageTexture.filterMode = cudaFilterModePoint;
	InputImageTexture.normalized = false;
	cutilSafeCall(cudaBindTextureToArray(InputImageTexture, InputImageArray));
	
	// Kernel Invocation
	dim3 DimBlock(16, 16);
	dim3 DimGrid(DivUp(w, DimBlock.x), DivUp(h, DimBlock.y));
	CUDARangeToWorldKernel<16,16><<<DimGrid,DimBlock>>>(w, h, duplicate);

	// Unbind texture
	cutilSafeCall(cudaUnbindTexture(InputImageTexture));

	CUT_CHECK_ERROR("Kernel execution failed");
}

extern "C"
void
CUDAFindLandmarks(float4* source, float4* target, float4* source_out, float4* target_out, int* indices_source, int* indices_target, int numLandmarks)
{	
	// Kernel Invocation
	CUDAFindLandmarksKernel<<<DivUp(numLandmarks, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(source, target, source_out, target_out, indices_source, indices_target, numLandmarks);

	CUT_CHECK_ERROR("Kernel execution failed");
}

extern "C"
void
CUDAFindNNBF(float4* source, float4* target, int* correspondences, int numLandmarks)
{
	// Kernel Invocation
	CUDAFindNNBFKernel<<<DivUp(numLandmarks, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(source, target, correspondences, numLandmarks);

	CUT_CHECK_ERROR("Kernel execution failed");
}

extern "C"
void
CUDATransfromLandmarks(float4* toBeTransformed, double matrix[4][4], int numLandmarks)
{
	// allocate memory for transformation matrix (will be stored linearly) and copy it
	float tmp[16];
	tmp[0] = (float)matrix[0][0];
	tmp[1] = (float)matrix[0][1];
	tmp[2] = (float)matrix[0][2];
	tmp[3] = (float)matrix[0][3];
	tmp[4] = (float)matrix[1][0];
	tmp[5] = (float)matrix[1][1];
	tmp[6] = (float)matrix[1][2];
	tmp[7] = (float)matrix[1][3];
	tmp[8] = (float)matrix[2][0];
	tmp[9] = (float)matrix[2][1];
	tmp[10] = (float)matrix[2][2];
	tmp[11] = (float)matrix[2][3];
	tmp[12] = (float)matrix[3][0];
	tmp[13] = (float)matrix[3][1];
	tmp[14] = (float)matrix[3][2];
	tmp[15] = (float)matrix[3][3];
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_matrix, tmp, 16*sizeof(float)));
	
	// transform points
	CUDATransformLandmarksKernel<<<DivUp(numLandmarks, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(toBeTransformed);
	CUT_CHECK_ERROR("Kernel execution failed (while transforming points)");
}