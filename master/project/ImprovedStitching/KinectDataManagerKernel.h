#ifndef KINECTDATAMANAGERKERNEL_H__
#define KINECTDATAMANAGERKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <channel_descriptor.h>

#include "defs.h"
#include "float.h"

// Texture that holds the input range in order to convert to world coordinates
texture<float, 2, cudaReadModeElementType> InputImageTexture;


//----------------------------------------------------------------------------
template<unsigned int BlockSizeX, unsigned int BlockSizeY>
__global__ void CUDARangeToWorldKernel(float* pointsOut)
{
	// 2D index and linear index within this thread block
	int tu = threadIdx.x;
	int tv = threadIdx.y;

	// Global 2D index and linear index.
	float gu = blockIdx.x*BlockSizeX+tu;
	float gv = blockIdx.y*BlockSizeY+tv;

	// Check for out-of-bounds
	if ( gu >= KINECT_IMAGE_WIDTH || gv >= KINECT_IMAGE_HEIGHT )
		return;

	// The range value
	float value = tex2D(InputImageTexture, gu, gv);

	if ( value < 500.f )
		value = sqrtf(-1.0f);

	float X2Z = 1.209f;
	float Y2Z = 0.9132f;
	float fNormalizedX = gu / KINECT_IMAGE_WIDTH - 0.5f; // Check for float
	float x = fNormalizedX * value * X2Z;

	float fNormalizedY = 0.5f - gv / KINECT_IMAGE_HEIGHT;
	float y = fNormalizedY * value * Y2Z;

	// Set the WC for the points without Mesh Structure (beware: optimized, works only for 6D data, where the first 3 components are x, y and z!)
	float3* PointsOutFloat3 = (float3*)pointsOut;
	PointsOutFloat3[(int)(gv*KINECT_IMAGE_WIDTH + gu)*2] = make_float3(x, y, value);
}


//----------------------------------------------------------------------------
__global__ void kernelExtractLandmarks(float* landmarksOut, float* pointsIn, unsigned long* indices, unsigned long numLandmarks)
{
	// Compute tid
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= numLandmarks)
		return;

	// Search for a valid position heuristically
	unsigned long PointsIdx = indices[tid]*ICP_DATA_DIM;
	unsigned long LandmarksIdx = tid*2;
	int FailCounter = 0;
	while (pointsIn[PointsIdx] != pointsIn[PointsIdx] && ++FailCounter < 128)
		PointsIdx = (PointsIdx + 32*ICP_DATA_DIM) % (KINECT_IMAGE_WIDTH * KINECT_IMAGE_HEIGHT * ICP_DATA_DIM);

	// Write landmarks using float3 for speedup
	float3* LandmarksOutFloat3 = (float3*)landmarksOut;
	float3* PointsInFloat3 = (float3*)pointsIn;

	LandmarksOutFloat3[LandmarksIdx] = PointsInFloat3[PointsIdx/3];
	LandmarksOutFloat3[LandmarksIdx+1] = PointsInFloat3[PointsIdx/3+1];
}


#endif // KINECTDATAMANAGERKERNEL_H__