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
	float fNormalizedX = gu / KINECT_IMAGE_WIDTH - 0.5f; // check for float
	float x = fNormalizedX * value * X2Z;

	float fNormalizedY = 0.5f - gv / KINECT_IMAGE_HEIGHT;
	float y = fNormalizedY * value * Y2Z;

	// The corresponding x,y,z triple (world coordinates)
	//float4 WC = make_float4(x, y, value, 1.0f);

	// Set the WC for the points without Mesh Structure
	pointsOut[(int)(gv*KINECT_IMAGE_WIDTH + gu)*ICP_DATA_DIM+0] = x;
	pointsOut[(int)(gv*KINECT_IMAGE_WIDTH + gu)*ICP_DATA_DIM+1] = y;
	pointsOut[(int)(gv*KINECT_IMAGE_WIDTH + gu)*ICP_DATA_DIM+2] = value;
}


//----------------------------------------------------------------------------
__global__ void kernelExtractLandmarks(float* landmarksOut, float* pointsIn, unsigned long* indices, unsigned long numLandmarks)
{
	// Compute tid
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= numLandmarks)
		return;

	// Search for a valid position
	unsigned long PointsIdx = indices[tid]*ICP_DATA_DIM;
	unsigned long LandmarksIdx = tid*ICP_DATA_DIM;
	int FailCounter = 0;
	while (pointsIn[PointsIdx+0] != pointsIn[PointsIdx+0] && ++FailCounter < 128)
		PointsIdx = (PointsIdx + 32*ICP_DATA_DIM) % (KINECT_IMAGE_WIDTH * KINECT_IMAGE_HEIGHT * ICP_DATA_DIM);

	// Write landmarks
	landmarksOut[LandmarksIdx+0] = pointsIn[PointsIdx+0]; // X
	landmarksOut[LandmarksIdx+1] = pointsIn[PointsIdx+1]; // Y
	landmarksOut[LandmarksIdx+2] = pointsIn[PointsIdx+2]; // Z
		
	// Normalize color
	float r = pointsIn[PointsIdx+3];
	float g = pointsIn[PointsIdx+4];
	float b = pointsIn[PointsIdx+5];

	//float rgb = r+g+b;
	const float rgb = 1.f;

	landmarksOut[LandmarksIdx+3] = r/(rgb);
	landmarksOut[LandmarksIdx+4] = g/(rgb);
	landmarksOut[LandmarksIdx+5] = b/(rgb);
}


#endif // KINECTDATAMANAGERKERNEL_H__