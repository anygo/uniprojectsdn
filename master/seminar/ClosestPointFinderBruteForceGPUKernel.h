#ifndef ClosestPointFinderBruteForceGPUKernel_H__
#define ClosestPointFinderBruteForceGPUKernel_H__

#include "defs.h"
#include "float.h"

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>

// global pointers for gpu... 
 __device__ unsigned short* dev_indices;
 __device__ PointCoords* dev_sourceCoords;
 __device__ PointColors* dev_sourceColors;
 __device__ PointCoords* dev_targetCoords;
 __device__ PointColors* dev_targetColors;
 __device__ float* dev_distances;
 __device__ __constant__ float dev_transformationMatrix[16];



__global__
void kernelWithRGB(int nrOfPoints, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, PointCoords* targetCoords, PointColors* targetColors, float* distances) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	unsigned short idx;
	float spaceDist;
	float colorDist;
	float dist;
	float x_dist, y_dist, z_dist;
	float r_dist, g_dist, b_dist;

	switch (metric)
	{
	case ABSOLUTE_DISTANCE:
		for (int i = 0; i < nrOfPoints; ++i)
		{
			x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			y_dist = sourceCoords[tid].y - targetCoords[i].y;
			z_dist = sourceCoords[tid].z - targetCoords[i].z;
			spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist);

			// always use euclidean distance for colors...
			r_dist = sourceColors[tid].r - targetColors[i].r; 
			g_dist = sourceColors[tid].g - targetColors[i].g;
			b_dist = sourceColors[tid].b - targetColors[i].b;
			colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
			dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;
			if (dist < minDist)
			{
				minDist = dist;
				idx = i;
			}
		} 
		break;
	case LOG_ABSOLUTE_DISTANCE:
		for (int i = 0; i < nrOfPoints; ++i)
		{
			x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			y_dist = sourceCoords[tid].y - targetCoords[i].y;
			z_dist = sourceCoords[tid].z - targetCoords[i].z;
			spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f);

			// always use euclidean distance for colors...
			r_dist = sourceColors[tid].r - targetColors[i].r; 
			g_dist = sourceColors[tid].g - targetColors[i].g;
			b_dist = sourceColors[tid].b - targetColors[i].b;
			colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
			dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;
			if (dist < minDist)
			{
				minDist = dist;
				idx = i;
			}
		} 
		break;
	case SQUARED_DISTANCE:
		for (int i = 0; i < nrOfPoints; ++i)
		{
			x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			y_dist = sourceCoords[tid].y - targetCoords[i].y;
			z_dist = sourceCoords[tid].z - targetCoords[i].z;
			spaceDist = ((x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist));

			// always use euclidean distance for colors...
			r_dist = sourceColors[tid].r - targetColors[i].r; 
			g_dist = sourceColors[tid].g - targetColors[i].g;
			b_dist = sourceColors[tid].b - targetColors[i].b;
			colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
			dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;
			if (dist < minDist)
			{
				minDist = dist;
				idx = i;
			}
		} 
		break;
	}

	distances[tid] = minDist;
	indices[tid] = idx;
}

__global__
void kernelWithoutRGB(int nrOfPoints, int metric, unsigned short* indices, PointCoords* sourceCoords, PointCoords* targetCoords, float* distances) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	unsigned short idx;
	float spaceDist;
	float x_dist, y_dist, z_dist;

	switch (metric)
	{
	case ABSOLUTE_DISTANCE:
		for (int i = 0; i < nrOfPoints; ++i)
		{
			x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			y_dist = sourceCoords[tid].y - targetCoords[i].y;
			z_dist = sourceCoords[tid].z - targetCoords[i].z;
			spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist);
			if (spaceDist < minDist)
			{
				minDist = spaceDist;
				idx = i;
			}
		} 
		break;
	case LOG_ABSOLUTE_DISTANCE:
		for (int i = 0; i < nrOfPoints; ++i)
		{
			x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			y_dist = sourceCoords[tid].y - targetCoords[i].y;
			z_dist = sourceCoords[tid].z - targetCoords[i].z;
			spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f);
			if (spaceDist < minDist)
			{
				minDist = spaceDist;
				idx = i;
			}
		} 
		break;
	case SQUARED_DISTANCE:
		for (int i = 0; i < nrOfPoints; ++i)
		{
			x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			y_dist = sourceCoords[tid].y - targetCoords[i].y;
			z_dist = sourceCoords[tid].z - targetCoords[i].z;
			spaceDist = ((x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist));
			if (spaceDist < minDist)
			{
				minDist = spaceDist;
				idx = i;
			}
		} 
		break;
	}

	distances[tid] = minDist;
	indices[tid] = idx;
}


__global__
void kernelTransformPointsAndComputeDistance(PointCoords* sourceCoords, float* distances)
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	// compute homogeneous transformation
	float x = dev_transformationMatrix[0]*sourceCoords[tid].x + dev_transformationMatrix[1]*sourceCoords[tid].y + dev_transformationMatrix[2]*sourceCoords[tid].z + dev_transformationMatrix[3];
	float y = dev_transformationMatrix[4]*sourceCoords[tid].x + dev_transformationMatrix[5]*sourceCoords[tid].y + dev_transformationMatrix[6]*sourceCoords[tid].z + dev_transformationMatrix[7];
	float z = dev_transformationMatrix[8]*sourceCoords[tid].x + dev_transformationMatrix[9]*sourceCoords[tid].y + dev_transformationMatrix[10]*sourceCoords[tid].z + dev_transformationMatrix[11];
	float w = dev_transformationMatrix[12]*sourceCoords[tid].x + dev_transformationMatrix[13]*sourceCoords[tid].y + dev_transformationMatrix[14]*sourceCoords[tid].z + dev_transformationMatrix[15];

	// divide by the last component
	x = x/w;
	y = y/w;
	z = z/w;

	// compute distance to previous point
	distances[tid] = (sourceCoords[tid].x - x)*(sourceCoords[tid].x - x) + (sourceCoords[tid].y - y)*(sourceCoords[tid].y - y) + (sourceCoords[tid].z - z)*(sourceCoords[tid].z - z);

	// set new coordinates
	sourceCoords[tid].x = x;
	sourceCoords[tid].y = y;
	sourceCoords[tid].z = z;
}


#endif // ClosestPointFinderBruteForceGPUKernel_H__