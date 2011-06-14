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
unsigned short* dev_indices;
PointCoords* dev_sourceCoords;
__constant__ PointColors* dev_sourceColors;
__constant__ PointCoords* dev_targetCoords;
__constant__ PointColors* dev_targetColors;
unsigned short* dev_representatives;
unsigned short* dev_pointToRep;

RepGPU* dev_repsGPU;

float* dev_distances;
__constant__ float dev_transformationMatrix[16];





///////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////
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





///////////////////////////////////////////////////////////////////////////////
// Brute Force
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelWithRGBBruteForce(int nrOfPoints, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, PointCoords* targetCoords, PointColors* targetColors, float* distances) 
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
void kernelWithoutRGBBruteForce(int nrOfPoints, int metric, unsigned short* indices, PointCoords* sourceCoords, PointCoords* targetCoords, float* distances) 
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





///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelRBC(int nrOfPoints, int nrOfReps, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, PointCoords* targetCoords, PointColors* targetColors, float* distances, unsigned short* representatives, unsigned short* pointToRep) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	unsigned short nearestRepresentative = 0;

	// step 1: search nearest representative
	for (int i = 0; i < nrOfReps; ++i)
	{
		float x_dist = sourceCoords[tid].x - targetCoords[representatives[i]].x; 
		float y_dist = sourceCoords[tid].y - targetCoords[representatives[i]].y;
		float z_dist = sourceCoords[tid].z - targetCoords[representatives[i]].z;
		float spaceDist;

		switch (metric)
		{
		case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
		}


		// always use euclidean distance for colors...
		float r_dist = sourceColors[tid].r - targetColors[representatives[i]].r; 
		float g_dist = sourceColors[tid].g - targetColors[representatives[i]].g;
		float b_dist = sourceColors[tid].b - targetColors[representatives[i]].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
		float dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;

		if (dist < minDist)
		{
			minDist = dist;
			nearestRepresentative = i;
		}
	}


	// step 2: search nearest neighbor in list of representatives
	minDist = FLT_MAX;
	int nearestNeighborIndex = 0;

	for (int i = 0; i < nrOfPoints; ++i)
	{
		if (pointToRep[i] == nearestRepresentative)
		{
			float x_dist = sourceCoords[tid].x - targetCoords[i].x; 
			float y_dist = sourceCoords[tid].y - targetCoords[i].y;
			float z_dist = sourceCoords[tid].z - targetCoords[i].z;
			float spaceDist;

			switch (metric)
			{
			case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
			case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
			case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
			}

			// always use euclidean distance for colors...
			float r_dist = sourceColors[tid].r - targetColors[i].r; 
			float g_dist = sourceColors[tid].g - targetColors[i].g;
			float b_dist = sourceColors[tid].b - targetColors[i].b;
			float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
			float dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;

			if (dist < minDist)
			{
				minDist = dist;
				nearestNeighborIndex = i;
			}
		}
	}

	distances[tid] = minDist;
	indices[tid] = nearestNeighborIndex;
}
///////////////////////////////////////////////////////////////////////////////
// Random Ball Cover 2
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelRBC2(int nrOfPoints, int nrOfReps, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, PointCoords* targetCoords, PointColors* targetColors, float* distances, RepGPU* dev_repsGPU) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	int nearestRepresentative;

	// step 1: search nearest representative
	for (int i = 0; i < nrOfReps; ++i)
	{
		float x_dist = sourceCoords[tid].x - targetCoords[dev_repsGPU[i].index].x; 
		float y_dist = sourceCoords[tid].y - targetCoords[dev_repsGPU[i].index].y;
		float z_dist = sourceCoords[tid].z - targetCoords[dev_repsGPU[i].index].z;
		float spaceDist;

		switch (metric)
		{
		case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
		}


		// always use euclidean distance for colors...
		float r_dist = sourceColors[tid].r - targetColors[dev_repsGPU[i].index].r; 
		float g_dist = sourceColors[tid].g - targetColors[dev_repsGPU[i].index].g;
		float b_dist = sourceColors[tid].b - targetColors[dev_repsGPU[i].index].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
		float dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;

		if (dist < minDist)
		{
			minDist = dist;
			nearestRepresentative = i;
		}
	}

	// step 2: search nearest neighbor in list of representative
	minDist = FLT_MAX;
	int nearestNeighborIndex = 0;
	for (int i = 0; i < dev_repsGPU[nearestRepresentative].nrOfPoints; ++i)
	{
		float x_dist = sourceCoords[tid].x - targetCoords[dev_repsGPU[nearestRepresentative].dev_points[i]].x; 
		float y_dist = sourceCoords[tid].y - targetCoords[dev_repsGPU[nearestRepresentative].dev_points[i]].y;
		float z_dist = sourceCoords[tid].z - targetCoords[dev_repsGPU[nearestRepresentative].dev_points[i]].z;
		float spaceDist;

		switch (metric)
		{
		case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		case SQUARED_DISTANCE: spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); break;
		}


		// always use euclidean distance for colors...
		float r_dist = sourceColors[tid].r - targetColors[dev_repsGPU[nearestRepresentative].dev_points[i]].r; 
		float g_dist = sourceColors[tid].g - targetColors[dev_repsGPU[nearestRepresentative].dev_points[i]].g;
		float b_dist = sourceColors[tid].b - targetColors[dev_repsGPU[nearestRepresentative].dev_points[i]].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
		float dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;

		if (dist < minDist)
		{
			minDist = dist;
			nearestNeighborIndex = dev_repsGPU[nearestRepresentative].dev_points[i];
		}
	}

	distances[tid] = minDist;
	indices[tid] = nearestNeighborIndex;
}

#endif // StitchingPluginKernel_H__