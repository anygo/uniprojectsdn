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
__constant__ RepGPU dev_repsGPU[MAX_REPRESENTATIVES];



///////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelTransformPointsAndComputeDistance()
{
	unsigned int tid = blockIdx.x;

	// compute homogeneous transformation
	float x =
		dev_transformationMatrix[0] * dev_conf->sourceCoords[tid].x +
		dev_transformationMatrix[1] * dev_conf->sourceCoords[tid].y +
		dev_transformationMatrix[2] * dev_conf->sourceCoords[tid].z +
		dev_transformationMatrix[3];
	float y =
		dev_transformationMatrix[4] * dev_conf->sourceCoords[tid].x +
		dev_transformationMatrix[5] * dev_conf->sourceCoords[tid].y +
		dev_transformationMatrix[6] * dev_conf->sourceCoords[tid].z +
		dev_transformationMatrix[7];
	float z =
		dev_transformationMatrix[8] * dev_conf->sourceCoords[tid].x +
		dev_transformationMatrix[9] * dev_conf->sourceCoords[tid].y +
		dev_transformationMatrix[10] * dev_conf->sourceCoords[tid].z +
		dev_transformationMatrix[11];
	float w =
		dev_transformationMatrix[12] * dev_conf->sourceCoords[tid].x +
		dev_transformationMatrix[13] * dev_conf->sourceCoords[tid].y +
		dev_transformationMatrix[14] * dev_conf->sourceCoords[tid].z +
		dev_transformationMatrix[15];

	// divide by the last component
	x /= w;
	y /= w;
	z /= w;

	// compute distance to previous point
	dev_conf->distances[tid] =
		(dev_conf->sourceCoords[tid].x - x) * (dev_conf->sourceCoords[tid].x - x) +
		(dev_conf->sourceCoords[tid].y - y) * (dev_conf->sourceCoords[tid].y - y) +
		(dev_conf->sourceCoords[tid].z - z) * (dev_conf->sourceCoords[tid].z - z);

	// set new coordinates
	dev_conf->sourceCoords[tid].x = x;
	dev_conf->sourceCoords[tid].y = y;
	dev_conf->sourceCoords[tid].z = z;
}

__device__
float kernelComputeDistanceSourceTarget(int idx1, int idx2)
{
		float x_dist = dev_conf->sourceCoords[idx1].x - dev_conf->targetCoords[idx2].x; 
		float y_dist = dev_conf->sourceCoords[idx1].y - dev_conf->targetCoords[idx2].y;
		float z_dist = dev_conf->sourceCoords[idx1].z - dev_conf->targetCoords[idx2].z;
		float spaceDist;

		//switch (dev_conf->metric)
		//{
		//case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		//case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		//case SQUARED_DISTANCE: 
			spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); //break;
		//}


		// always use euclidean distance for colors...
		float r_dist = dev_conf->sourceColors[idx1].r - dev_conf->targetColors[idx2].r; 
		float g_dist = dev_conf->sourceColors[idx1].g - dev_conf->targetColors[idx2].g;
		float b_dist = dev_conf->sourceColors[idx1].b - dev_conf->targetColors[idx2].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	
		return (1 - dev_conf->weightRGB) * spaceDist + dev_conf->weightRGB * colorDist;
}
__device__
float kernelComputeDistanceTargetTarget(int idx1, int idx2)
{
		float x_dist = dev_conf->targetCoords[idx1].x - dev_conf->targetCoords[idx2].x; 
		float y_dist = dev_conf->targetCoords[idx1].y - dev_conf->targetCoords[idx2].y;
		float z_dist = dev_conf->targetCoords[idx1].z - dev_conf->targetCoords[idx2].z;
		float spaceDist;

		//switch (dev_conf->metric)
		//{
		//case ABSOLUTE_DISTANCE: spaceDist = abs(x_dist) + abs(y_dist) + abs(z_dist); break;
		//case LOG_ABSOLUTE_DISTANCE: spaceDist = log(abs(x_dist) + abs(y_dist) + abs(z_dist) + 1.f); break;
		//case SQUARED_DISTANCE: 
			spaceDist = (x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist); //break;
		//}


		// always use euclidean distance for colors...
		float r_dist = dev_conf->targetColors[idx1].r - dev_conf->targetColors[idx2].r; 
		float g_dist = dev_conf->targetColors[idx1].g - dev_conf->targetColors[idx2].g;
		float b_dist = dev_conf->targetColors[idx1].b - dev_conf->targetColors[idx2].b;
		float colorDist = (r_dist * r_dist) + (g_dist * g_dist) + (b_dist * b_dist);
	
		return (1 - dev_conf->weightRGB) * spaceDist + dev_conf->weightRGB * colorDist;
}
///////////////////////////////////////////////////////////////////////////////
// Brute Force
///////////////////////////////////////////////////////////////////////////////
__global__
void kernelBruteForce() 
{
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	unsigned short idx;

	for (int i = 0; i < dev_conf->nrOfPoints; ++i)
	{
		float dist = kernelComputeDistanceSourceTarget(tid, i);
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
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	int nearestRepresentative;

	// step 1: search nearest representative
	for (int i = 0; i < nrOfReps; ++i)
	{
		float dist = kernelComputeDistanceSourceTarget(tid, dev_repsGPU[i].index);

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
		float dist = kernelComputeDistanceSourceTarget(tid, dev_repsGPU[nearestRepresentative].dev_points[i]);
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
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;
	int nearestRepresentative;

	// step 1: search nearest representative
	for (int i = 0; i < nrOfReps; ++i)
	{
		float dist = kernelComputeDistanceTargetTarget(tid, reps[i]);

		if (dist < minDist)
		{
			minDist = dist;
			nearestRepresentative = i;
		}
	}

	pointToRep[tid] = nearestRepresentative;
}

#endif // StitchingPluginKernel_H__