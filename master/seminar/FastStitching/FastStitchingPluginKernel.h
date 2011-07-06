#ifndef FastStitchingPluginKernel_H__
#define FastStitchingPluginKernel_H__

#include "defs.h"
#include "float.h"

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>


// global pointers for gpu... 
__constant__ float devWeightRGB[1];

__constant__ float dev_transformationMatrix[16];

// RBC
__constant__ unsigned short* dev_representatives;
__constant__ unsigned short* dev_pointToRep;
__constant__ unsigned short* dev_reps;
__constant__ RepGPU dev_repsGPU[MAX_REPRESENTATIVES];

// Texture that holds the input range in order to convert to world coordinates
texture<float, 2, cudaReadModeElementType> InputImageTexture;


///////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////
template<unsigned int BlockSizeX, unsigned int BlockSizeY>
__global__ void
CUDARangeToWorldKernel(float4* duplicate)
{
	// 2D index and linear index within this thread block
	int tu = threadIdx.x;
	int tv = threadIdx.y;

	// Global 2D index and linear index.
	float gu = blockIdx.x*BlockSizeX+tu;
	float gv = blockIdx.y*BlockSizeY+tv;

	// Check for out-of-bounds
	if ( gu >= FRAME_SIZE_X || gv >= FRAME_SIZE_Y )
		return;

	// The range value
	float value = tex2D(InputImageTexture, gu, gv);

	// The corresponding x,y,z triple
	float4 WC;

	if ( value < 500.f )
		value = sqrtf(-1.0f);

	float X2Z = 1.209f;
	float Y2Z = 0.9132f;
	float fNormalizedX = gu / FRAME_SIZE_X - 0.5f; // check for float
	float x = fNormalizedX * value * X2Z;

	float fNormalizedY = 0.5f - gv / FRAME_SIZE_Y;
	float y = fNormalizedY * value * Y2Z;

	// World coordinates
	WC = make_float4(x, y, value, 1.0f);

	// Set the WC for the duplicate without Mesh Structure
	duplicate[(int)(gv*FRAME_SIZE_X + gu)] = WC;
}

__global__
void kernelTransformPointsAndComputeDistance(float4* points, float* distances, int numPoints)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float m[16];

	if (threadIdx.x < 16)
		m[threadIdx.x] = dev_transformationMatrix[threadIdx.x];

	__syncthreads();

	if (tid >= numPoints)
		return;

	float4 p = points[tid];

	// compute homogeneous transformation
	float x = m[0]  * p.x + m[1]  * p.y + m[2]  * p.z + m[3];
	float y = m[4]  * p.x + m[5]  * p.y + m[6]  * p.z + m[7];
	float z = m[8]  * p.x + m[9]  * p.y + m[10] * p.z + m[11];
	float w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15];

	// divide by the last component
	x /= w;
	y /= w;
	z /= w;

	// set new coordinates
	points[tid] = make_float4(x, y, z, 1.f);

	if (!distances)
		return;
	
	// compute distance to previous point
	float xDiff = p.x - x;
	float yDiff = p.y - y;
	float zDiff = p.z - z;
	distances[tid] = xDiff*xDiff + yDiff*yDiff + zDiff*zDiff;
}

__global__
void kernelExtractLandmarks(float4* devWCsIn, uchar3* devColorsIn, unsigned int* devIndicesIn, float4* devLandmarksOut, float4* devColorsOut)
{
	// get source[tid] for this thread
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	int idx = devIndicesIn[tid];
	while (devWCsIn[idx].x != devWCsIn[idx].x)
		idx = (idx + 1) % (FRAME_SIZE_X * FRAME_SIZE_Y);
		
	// normalize color
	float r = devColorsIn[idx].x;
	float g = devColorsIn[idx].y;
	float b = devColorsIn[idx].z;

	devColorsOut[tid].x = r/(r+g+b);
	devColorsOut[tid].y = g/(r+g+b);
	devColorsOut[tid].z = b/(r+g+b);
	devColorsOut[tid].w = 1.f;

	devLandmarksOut[tid] = devWCsIn[idx];
}

__device__
float kernelComputeDistanceSourceTarget(float4* coords, float4* colors, float4* coords2, float4* colors2)
{
		float x_dist = coords->x - coords2->x; 
		float y_dist = coords->y - coords2->y;
		float z_dist = coords->z - coords2->z;
		float spaceDist;

		//spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist);
		//spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f);
		spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist);


		float r_dist = colors->x - colors2->x; 
		float g_dist = colors->y - colors2->y;
		float b_dist = colors->z - colors2->z;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	
		return (1 - devWeightRGB[0]) * spaceDist + devWeightRGB[0] * colorDist;
}



///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelRBC(int nrOfReps, unsigned int* indices, float* distances, float4* targetCoords, float4* targetColors, float4* sourceCoords, float4* sourceColors)
{
	// get source[tid] for this thread
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//if (tid >= dev_conf->nrOfPoints)
	//	return;

	float4 coords = sourceCoords[tid];
	float4 colors = sourceColors[tid];

	__shared__ float4 repCoordsBuffer[CUDA_BUFFER_SIZE];
	__shared__ float4 repColorsBuffer[CUDA_BUFFER_SIZE];

	float minDist = FLT_MAX;
	int nearestRepresentative;

	// step 1: search nearest representative
	for (int i = 0; i < nrOfReps; ++i)
	{
		if (i % CUDA_BUFFER_SIZE == 0)
		{
			__syncthreads();

			if (i % CUDA_BUFFER_SIZE == 0 && threadIdx.x < CUDA_BUFFER_SIZE && i + threadIdx.x < nrOfReps) 
			{
				unsigned int idx = dev_repsGPU[i + threadIdx.x].index;
				repCoordsBuffer[threadIdx.x] = targetCoords[idx];
				repColorsBuffer[threadIdx.x] = targetColors[idx];
			}
			__syncthreads();
		}

		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, &(repCoordsBuffer[i % CUDA_BUFFER_SIZE]), &(repColorsBuffer[i % CUDA_BUFFER_SIZE]));
		//float dist = kernelComputeDistanceSourceTarget(&coords, NULL, &(repCoordsBuffer[i % CUDA_BUFFER_SIZE]), NULL);

		if (dist < minDist)
		{
			minDist = dist;
			nearestRepresentative = i;
		}
	}

	// step 2: search nearest neighbor in list of representatives
	minDist = FLT_MAX;
	int nearestNeighborIndex = 0;
	for (int i = 0; i < dev_repsGPU[nearestRepresentative].nrOfPoints; ++i)
	{
		unsigned int idx = dev_repsGPU[nearestRepresentative].dev_points[i];
		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, &(targetCoords[idx]), &(targetColors[idx]));

		//float dist = kernelComputeDistanceSourceTarget(&coords, NULL, &(targetCoords[idx]), NULL);

		if (dist < minDist)
		{
			minDist = dist;
			nearestNeighborIndex = dev_repsGPU[nearestRepresentative].dev_points[i];
		}
	}

	distances[tid] = minDist;
	indices[tid] = nearestNeighborIndex;
}


__global__
void kernelPointsToReps(int nrOfReps, float4* targetCoords, float4* targetColors, unsigned int* indices, unsigned int* pointToRep)
{
	// get source[tid] for this thread
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	float4 coords = targetCoords[tid];
	float4 colors = targetColors[tid];

	__shared__ float4 repCoordsBuffer[CUDA_BUFFER_SIZE];
	__shared__ float4 repColorsBuffer[CUDA_BUFFER_SIZE];

	float minDist = FLT_MAX;
	unsigned int nearestRepresentative;

	// step 1: search nearest representative
	for (int i = 0; i < nrOfReps; ++i)
	{
		if (i % CUDA_BUFFER_SIZE == 0)
		{
			__syncthreads();
			if (i % CUDA_BUFFER_SIZE == 0 && threadIdx.x < CUDA_BUFFER_SIZE && i + threadIdx.x < nrOfReps) 
			{
				repCoordsBuffer[threadIdx.x] = targetCoords[indices[i + threadIdx.x]];
				repColorsBuffer[threadIdx.x] = targetColors[indices[i + threadIdx.x]];
			}
			__syncthreads();
		}

		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, &(repCoordsBuffer[i % CUDA_BUFFER_SIZE]), &(repColorsBuffer[i % CUDA_BUFFER_SIZE]));
		//float dist = kernelComputeDistanceSourceTarget(&coords, NULL, &(repCoordsBuffer[i % CUDA_BUFFER_SIZE]), NULL);

		if (dist < minDist)
		{
			minDist = dist;
			nearestRepresentative = i;
		}
	}

	pointToRep[tid] = nearestRepresentative;
}

#endif // FastStitchingPluginKernel_H__