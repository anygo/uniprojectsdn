#ifndef StitchingPluginKernel_H__
#define StitchingPluginKernel_H__

#include "defs.h"
#include "float.h"

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>


// global pointers for gpu... 
__constant__ float dev_transformationMatrix[16];
__constant__ GPUConfig dev_conf[1];
GPUConfig host_conf[1];

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
CUDARangeToWorldKernel(unsigned int NX, unsigned int NY, float4* duplicate)
{
	// 2D index and linear index within this thread block
	int tu = threadIdx.x;
	int tv = threadIdx.y;

	// Global 2D index and linear index.
	float gu = blockIdx.x*BlockSizeX+tu;
	float gv = blockIdx.y*BlockSizeY+tv;

	// Check for out-of-bounds
	if ( gu >= NX || gv >= NY )
		return;

	// The range value
	float value = tex2D(InputImageTexture, gu, gv);

	// The corresponding x,y,z triple
	float4 WC;

	if ( value < 500.f )
		value = sqrtf(-1.0f);

	float X2Z = 1.209f;
	float Y2Z = 0.9132f;
	float fNormalizedX = gu / NX - 0.5f; // check for float
	float x = fNormalizedX * value * X2Z;

	float fNormalizedY = 0.5f - gv / NY;
	float y = fNormalizedY * value * Y2Z;

	// World coordinates
	WC = make_float4(x, y, value, 1.0f);

	// Set the WC for the duplicate without Mesh Structure
	duplicate[(int)(gv*NX + gu)] = WC;
}

__global__
void kernelTransformPointsAndComputeDistance()
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float m[16];

	if (threadIdx.x < 16)
		m[threadIdx.x] = dev_transformationMatrix[threadIdx.x];

	__syncthreads();

	if (tid >= dev_conf->nrOfPoints)
		return;

	float xOld = dev_conf->sourceCoords[tid].x;
	float yOld = dev_conf->sourceCoords[tid].y;
	float zOld = dev_conf->sourceCoords[tid].z;

	// compute homogeneous transformation
	float x = m[0]  * xOld + m[1]  * yOld + m[2]  * zOld + m[3];
	float y = m[4]  * xOld + m[5]  * yOld + m[6]  * zOld + m[7];
	float z = m[8]  * xOld + m[9]  * yOld + m[10] * zOld + m[11];
	float w = m[12] * xOld + m[13] * yOld + m[14] * zOld + m[15];

	// divide by the last component
	x /= w;
	y /= w;
	z /= w;

	// compute distance to previous point
	float xDiff = xOld - x;
	float yDiff = yOld - y;
	float zDiff = zOld - z;
	dev_conf->distances[tid] = xDiff*xDiff + yDiff*yDiff + zDiff*zDiff;

	// set new coordinates
	dev_conf->sourceCoords[tid].x = x;
	dev_conf->sourceCoords[tid].y = y;
	dev_conf->sourceCoords[tid].z = z;
}

__device__
float kernelComputeDistanceSourceTarget(PointCoords* coords, PointColors* colors, PointCoords* coords2, PointColors* colors2)
{
		float x_dist = coords->x - coords2->x; 
		float y_dist = coords->y - coords2->y;
		float z_dist = coords->z - coords2->z;
		float spaceDist;

		//switch (dev_conf->metric)
		//{
		//case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		//case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		//case SQUARED_DISTANCE: 
			spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); //break;
		//}


		// always use euclidean distance for colors...
		float r_dist = colors->r - colors2->r; 
		float g_dist = colors->g - colors2->g;
		float b_dist = colors->b - colors2->b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	
		return (1 - dev_conf->weightRGB) * spaceDist + dev_conf->weightRGB * colorDist;
}

__device__
float kernelComputeDistanceSourceTarget(PointCoords* coords, PointColors* colors, int idx2)
{
		float x_dist = coords->x - dev_conf->targetCoords[idx2].x; 
		float y_dist = coords->y - dev_conf->targetCoords[idx2].y;
		float z_dist = coords->z - dev_conf->targetCoords[idx2].z;
		float spaceDist;

		//switch (dev_conf->metric)
		//{
		//case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		//case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		//case SQUARED_DISTANCE: 
			spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); //break;
		//}


		// always use euclidean distance for colors...
		float r_dist = colors->r - dev_conf->targetColors[idx2].r; 
		float g_dist = colors->g - dev_conf->targetColors[idx2].g;
		float b_dist = colors->b - dev_conf->targetColors[idx2].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	
		return (1 - dev_conf->weightRGB) * spaceDist + dev_conf->weightRGB * colorDist;
}

/*__device__
float kernelComputeDistanceTargetTarget(PointCoords* coords, PointColors* colors, int idx2)
{
		float x_dist = coords->x - dev_conf->targetCoords[idx2].x; 
		float y_dist = coords->y - dev_conf->targetCoords[idx2].y;
		float z_dist = coords->z - dev_conf->targetCoords[idx2].z;
		float spaceDist;

		//switch (dev_conf->metric)
		//{
		//case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		//case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		//case SQUARED_DISTANCE: 
			spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); //break;
		//}


		// always use euclidean distance for colors...
		float r_dist = colors->r - dev_conf->targetColors[idx2].r; 
		float g_dist = colors->g - dev_conf->targetColors[idx2].g;
		float b_dist = colors->b - dev_conf->targetColors[idx2].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	
		return (1 - dev_conf->weightRGB) * spaceDist + dev_conf->weightRGB * colorDist;
}
*/

///////////////////////////////////////////////////////////////////////////////
// Brute Force
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelBruteForce() 
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= dev_conf->nrOfPoints)
		return;

	PointCoords coords = dev_conf->sourceCoords[tid];
	PointColors colors = dev_conf->sourceColors[tid];

	float minDist = FLT_MAX;
	unsigned short idx;

	for (int i = 0; i < dev_conf->nrOfPoints; ++i)
	{
		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, i);
		if (dist < minDist)
		{
			minDist = dist;
			idx = i;
		}
	} 

	dev_conf->distances[tid] = minDist;
	dev_conf->indices[tid] = idx;
}

///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelRBC(int nrOfReps) 
{
	// get source[tid] for this thread
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= dev_conf->nrOfPoints)
		return;

	PointCoords coords = dev_conf->sourceCoords[tid];
	PointColors colors = dev_conf->sourceColors[tid];

	__shared__ PointCoords repCoordsBuffer[CUDA_BUFFER_SIZE];
	__shared__ PointColors repColorsBuffer[CUDA_BUFFER_SIZE];

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
				repCoordsBuffer[threadIdx.x] = dev_repsGPU[i + threadIdx.x].coords;
				repColorsBuffer[threadIdx.x] = dev_repsGPU[i + threadIdx.x].colors;
			}
			__syncthreads();
		}

		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, &(repCoordsBuffer[i % CUDA_BUFFER_SIZE]), &(repColorsBuffer[i % CUDA_BUFFER_SIZE]));

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
		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, dev_repsGPU[nearestRepresentative].dev_points[i]);
		if (dist < minDist)
		{
			minDist = dist;
			nearestNeighborIndex = dev_repsGPU[nearestRepresentative].dev_points[i];
		}
	}

	dev_conf->distances[tid] = minDist;
	dev_conf->indices[tid] = nearestNeighborIndex;
}

__global__
void kernelPointsToReps(int nrOfReps, unsigned short* pointToRep, unsigned short* reps)
{
	// get source[tid] for this thread
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= dev_conf->nrOfPoints)
		return;

	PointCoords coords = dev_conf->targetCoords[tid];
	PointColors colors = dev_conf->targetColors[tid];

	__shared__ PointCoords repCoordsBuffer[CUDA_BUFFER_SIZE];
	__shared__ PointColors repColorsBuffer[CUDA_BUFFER_SIZE];

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
				repCoordsBuffer[threadIdx.x] = dev_conf->targetCoords[reps[i + threadIdx.x]];
				repColorsBuffer[threadIdx.x] = dev_conf->targetColors[reps[i + threadIdx.x]];
			}
			__syncthreads();
		}

		float dist = kernelComputeDistanceSourceTarget(&coords, &colors, &(repCoordsBuffer[i % CUDA_BUFFER_SIZE]), &(repColorsBuffer[i % CUDA_BUFFER_SIZE]));
		//float dist = kernelComputeDistanceSourceTarget(&coords, &colors, &repCoordsBuffer, &repColorsBuffer);

		if (dist < minDist)
		{
			minDist = dist;
			nearestRepresentative = i;
		}
	}

	pointToRep[tid] = nearestRepresentative;
}

#endif // StitchingPluginKernel_H__