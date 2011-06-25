#include "FastStitchingKernel.h"
#include <cutil.h>

#define CUDA_THREADS_PER_BLOCK 128

extern "C"
void
CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray, float4 *DeviceOutput, int w, int h, float fx, float fy, float cx, float cy, float k1, float k2)
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
	CUDARangeToWorldKernel<16,16><<<DimGrid,DimBlock>>>(w, h, DeviceOutput, duplicate, fx, fy, cx, cy, k1, k2);

	// Copy the current WCs into duplicate
	//cutilSafeCall(cudaMemcpy(duplicate, DeviceOutput, sizeof(float4)*640*480, cudaMemcpyDeviceToDevice));	

	// Unbind texture
	cutilSafeCall(cudaUnbindTexture(InputImageTexture));

	CUT_CHECK_ERROR("Kernel execution failed");
}

extern "C"
void
CUDANearestNeighborBF(float4* source, float4* target, float4* source_out, float4* target_out, int* indices, int numLandmarks)
{	
	// Kernel Invocation
	CUDANearestNeighborBFKernel<<<DivUp(numLandmarks, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(source, target, source_out, target_out, indices, numLandmarks);

	CUT_CHECK_ERROR("Kernel execution failed");
}
