#include "FastStitchingKernel.h"
#include <cutil.h>


extern "C"
void
CUDARangeToWorld(const cudaArray *InputImageArray, float4 *DeviceOutput, int w, int h, float fx, float fy, float cx, float cy, float k1, float k2)
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
	CUDARangeToWorldKernel<16,16><<<DimGrid,DimBlock>>>(w, h, DeviceOutput, fx, fy, cx, cy, k1, k2);

	// Unbind texture
	cutilSafeCall(cudaUnbindTexture(InputImageTexture));

	CUT_CHECK_ERROR("Kernel execution failed");
}

extern "C"
void
TEST(float4* DeviceOutput)
{
	// Kernel Invocation
	dim3 DimBlock(16, 16);
	dim3 DimGrid(DivUp(640, DimBlock.x), DivUp(480, DimBlock.y));
	TESTKernel<16,16><<<DimGrid,DimBlock>>>(DeviceOutput);
}

