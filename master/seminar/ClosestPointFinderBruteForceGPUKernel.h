#ifndef ClosestPointFinderBruteForceGPUKernel_H__
#define ClosestPointFinderBruteForceGPUKernel_H__

#include "defs.h"
#include "float.h"

__global__
void kernelWithoutRGB(int nrOfPoints, int metric, int* indices, Point6D* source, Point6D* target) 
{
	// get source[tid] for this thread
	int tid = blockIdx.x;

	float minDist = FLT_MAX;

	int idx = -1;


	for (int i = 0; i < nrOfPoints; ++i)
	{
		float spaceDist = 0.f;

		/*switch (metric)
		{
		case ABSOLUTE_DISTANCE:
			spaceDist = std::abs(source[tid].x - target[i].x) + std::abs(source[tid].y - target[i].y) + std::abs(source[tid].z - target[i].z); break;
		case LOG_ABSOLUTE_DISTANCE:
			spaceDist = std::log(std::abs(source[tid].x - target[i].x) + std::abs(source[tid].y - target[i].y) + std::abs(source[tid].z - target[i].z) + 1.0); break;
		case SQUARED_DISTANCE:*/
			spaceDist = ((source[tid].x - target[i].x)*(source[tid].x - target[i].x) + (source[tid].y - target[i].y)*(source[tid].y - target[i].y) + (source[tid].z - target[i].z)*(source[tid].z - target[i].z));
		//}

		spaceDist;

		if (spaceDist < minDist)
		{
			minDist = spaceDist;
			idx = i;
		}
	}

	indices[tid] = idx;
}

__global__
void kernelWithRGB(int nrOfPoints, int metric, float weightRGB, int* indices, Point6D* source, Point6D* target) 
{
	// get source[tid] for this thread
	int tid = blockIdx.x;

	float minDist = FLT_MAX;

	int idx = -1;


	for (int i = 0; i < nrOfPoints; ++i)
	{
		float spaceDist = 0.f;
		float colorDist = 0.f;

		/*switch (metric)
		{
		case ABSOLUTE_DISTANCE:
			spaceDist = std::abs(source[tid].x - target[i].x) + std::abs(source[tid].y - target[i].y) + std::abs(source[tid].z - target[i].z); break;
		case LOG_ABSOLUTE_DISTANCE:
			spaceDist = std::log(std::abs(source[tid].x - target[i].x) + std::abs(source[tid].y - target[i].y) + std::abs(source[tid].z - target[i].z) + 1.0); break;
		case SQUARED_DISTANCE:*/
			spaceDist = ((source[tid].x - target[i].x)*(source[tid].x - target[i].x) + (source[tid].y - target[i].y)*(source[tid].y - target[i].y) + (source[tid].z - target[i].z)*(source[tid].z - target[i].z));
		//}

		/*switch (metric)
		{
		case ABSOLUTE_DISTANCE:
			colorDist = std::abs(source[tid].r - target[i].r) + std::abs(source[tid].g - target[i].g) + std::abs(source[tid].b - target[i].b); break;
		case LOG_ABSOLUTE_DISTANCE:
			colorDist = std::log(std::abs(source[tid].r - target[i].r) + std::abs(source[tid].g - target[i].g) + std::abs(source[tid].b - target[i].b) + 1.0); break;
		case SQUARED_DISTANCE:*/
			colorDist = ((source[tid].r - target[i].r)*(source[tid].r - target[i].r) + (source[tid].g - target[i].g)*(source[tid].g - target[i].g) + (source[tid].b - target[i].b)*(source[tid].b - target[i].b));
		//}

		
		float dist = (1 - weightRGB) * spaceDist + weightRGB * colorDist;

		if (dist < minDist)
		{
			minDist = dist;
			idx = i;
		}
	}

	indices[tid] = idx;
}


#endif // ClosestPointFinderBruteForceGPUKernel_H__