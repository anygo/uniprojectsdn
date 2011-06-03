#ifndef ClosestPointFinderBruteForceGPUKernel_H__
#define ClosestPointFinderBruteForceGPUKernel_H__

#include "defs.h"
#include "float.h"

__global__
void kernelWithRGB(int nrOfPoints, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, PointCoords* targetCoords, PointColors* targetColors) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;

	unsigned short idx = 0;


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
			spaceDist = ((sourceCoords[tid].x - targetCoords[i].x)*(sourceCoords[tid].x - targetCoords[i].x) + (sourceCoords[tid].y - targetCoords[i].y)*(sourceCoords[tid].y - targetCoords[i].y) + (sourceCoords[tid].z - targetCoords[i].z)*(sourceCoords[tid].z - targetCoords[i].z));
		//}

		/*switch (metric)
		{
		case ABSOLUTE_DISTANCE:
			colorDist = std::abs(source[tid].r - target[i].r) + std::abs(source[tid].g - target[i].g) + std::abs(source[tid].b - target[i].b); break;
		case LOG_ABSOLUTE_DISTANCE:
			colorDist = std::log(std::abs(source[tid].r - target[i].r) + std::abs(source[tid].g - target[i].g) + std::abs(source[tid].b - target[i].b) + 1.0); break;
		case SQUARED_DISTANCE:*/
			colorDist = ((sourceColors[tid].r - targetColors[i].r)*(sourceColors[tid].r - targetColors[i].r) + (sourceColors[tid].g - targetColors[i].g)*(sourceColors[tid].g - targetColors[i].g) + (sourceColors[tid].b - targetColors[i].b)*(sourceColors[tid].b - targetColors[i].b));
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

__global__
void kernelWithoutRGB(int nrOfPoints, int metric, unsigned short* indices, PointCoords* sourceCoords, PointCoords* targetCoords) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	float minDist = FLT_MAX;

	unsigned short idx = 0;


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
			spaceDist = ((sourceCoords[tid].x - targetCoords[i].x)*(sourceCoords[tid].x - targetCoords[i].x) + (sourceCoords[tid].y - targetCoords[i].y)*(sourceCoords[tid].y - targetCoords[i].y) + (sourceCoords[tid].z - targetCoords[i].z)*(sourceCoords[tid].z - targetCoords[i].z));
		//}

		if (spaceDist < minDist)
		{
			minDist = spaceDist;
			idx = i;
		}
	}

	indices[tid] = idx;
}

__global__
void kernelTransformPoints(PointCoords* sourceCoords, float* m, float* distances)
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	// compute homogeneous transformation
	float x = m[0]*sourceCoords[tid].x + m[1]*sourceCoords[tid].y + m[2]*sourceCoords[tid].z + m[3];
	float y = m[4]*sourceCoords[tid].x + m[5]*sourceCoords[tid].y + m[6]*sourceCoords[tid].z + m[7];
	float z = m[8]*sourceCoords[tid].x + m[9]*sourceCoords[tid].y + m[10]*sourceCoords[tid].z + m[11];
	float w = m[12]*sourceCoords[tid].x + m[13]*sourceCoords[tid].y + m[14]*sourceCoords[tid].z + m[15];

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