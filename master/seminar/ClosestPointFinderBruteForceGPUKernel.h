#ifndef ClosestPointFinderBruteForceGPUKernel_H__
#define ClosestPointFinderBruteForceGPUKernel_H__

#include "defs.h"
#include "float.h"

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>

texture<float, 1, cudaReadModeElementType> tex;

__global__
void kernelWithRGB(int nrOfPoints, int metric, float weightRGB, unsigned short* indices, PointCoords* sourceCoords, PointColors* sourceColors, PointCoords* targetCoords, PointColors* targetColors) 
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;
	//tid = blockIdx.x*blockDim.x + threadIdx.x;

	float minDist = FLT_MAX;

	unsigned short idx = 0;
	
	// fuckingly did it!! freaking texture crap 0.5 addition is damn important somehow :D
	/*float x = tex1Dfetch(tex, float(tid*3.f + 0.5f));
	float y = tex1Dfetch(tex, float(tid*3.f + 1.5f));
	float z = tex1Dfetch(tex, float(tid*3.f + 2.5f));*/

	for (int i = 0; i < nrOfPoints; ++i)
	{
		float spaceDist = 0.f;
		float colorDist = 0.f;

		//switch (metric)
		//{
		//case ABSOLUTE_DISTANCE:
		//	spaceDist = std::abs(sourceCoords[tid].x - targetCoords[i].x) + std::abs(sourceCoords[tid].y - targetCoords[i].y) + std::abs(sourceCoords[tid].z - targetCoords[i].z); break;
			//spaceDist = std::abs(sourceCoords[tid].x - x) + std::abs(sourceCoords[tid].y - y) + std::abs(sourceCoords[tid].z - z); break;
		//case LOG_ABSOLUTE_DISTANCE:
		//	spaceDist = std::log(std::abs(sourceCoords[tid].x - targetCoords[i].x) + std::abs(sourceCoords[tid].y - targetCoords[i].y) + std::abs(sourceCoords[tid].z - targetCoords[i].z) + 1.0); break;
			//spaceDist = std::log(std::abs(sourceCoords[tid].x - x) + std::abs(sourceCoords[tid].y - y) + std::abs(sourceCoords[tid].z - z) + 1.0); break;
		//case SQUARED_DISTANCE:
			spaceDist = ((sourceCoords[tid].x - targetCoords[i].x)*(sourceCoords[tid].x - targetCoords[i].x) + (sourceCoords[tid].y - targetCoords[i].y)*(sourceCoords[tid].y - targetCoords[i].y) + (sourceCoords[tid].z - targetCoords[i].z)*(sourceCoords[tid].z - targetCoords[i].z));
			//spaceDist = ((sourceCoords[tid].x - x)*(sourceCoords[tid].x - x) + (sourceCoords[tid].y - y)*(sourceCoords[tid].y - y) + (sourceCoords[tid].z -z)*(sourceCoords[tid].z - z));
		//}

		// always use euclidean distance for colors...
		colorDist = ((sourceColors[tid].r - targetColors[i].r)*(sourceColors[tid].r - targetColors[i].r) + (sourceColors[tid].g - targetColors[i].g)*(sourceColors[tid].g - targetColors[i].g) + (sourceColors[tid].b - targetColors[i].b)*(sourceColors[tid].b - targetColors[i].b));

		
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