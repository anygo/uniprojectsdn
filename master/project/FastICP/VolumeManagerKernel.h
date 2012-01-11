#ifndef VOLUMEMANAGERKERNEL_H__
#define VOLUMEMANAGERKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "defs.h"


//----------------------------------------------------------------------------
__global__ void kernelAddPointsToVolume(float* points, float* voxels, float* origin, unsigned long numPts, unsigned int dimSize, unsigned int spacing)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= dimSize*dimSize*dimSize ) return;

	// Compute x-, y- and z-boundaries	
	int Tmp = tid;
	int zUnits = Tmp % dimSize;
	Tmp /= dimSize;
	int yUnits = Tmp % dimSize;
	Tmp /= dimSize;
	int xUnits = Tmp % dimSize;

	float xMin = origin[0] + (xUnits  )*spacing;
	float xMax = origin[0] + (xUnits+1)*spacing;

	float yMin = origin[1] + (yUnits  )*spacing;
	float yMax = origin[1] + (yUnits+1)*spacing;

	float zMin = origin[2] + (zUnits  )*spacing;
	float zMax = origin[2] + (zUnits+1)*spacing;


	// For each point, determine if it lies inside the voxel handled by this thread
	float r = 0, g = 0, b = 0;
	float Counter = 0;
	for (int i = 0; i < numPts; ++i)
	{
		float x,y,z;
		x = points[i*ICP_DATA_DIM+0];
		y = points[i*ICP_DATA_DIM+1];
		z = points[i*ICP_DATA_DIM+2];

		if (x >= xMin && x < xMax &&
			y >= yMin && y < yMax &&
			z >= zMin && z < zMax)
		{
			r += points[i*ICP_DATA_DIM+3];
			g += points[i*ICP_DATA_DIM+4];
			b += points[i*ICP_DATA_DIM+5];
			++Counter;
		}
	}

	// If at least one point is inside this voxel, store the average color
	if (Counter > 0)
	{
		// Store colors within a 4-byte-block as uchars
		/*float Colors;
		unsigned char* ColorsUChar = (unsigned char*)&Colors;
		ColorsUChar[0] = r/Counter;
		ColorsUChar[1] = g/Counter;
		ColorsUChar[2] = b/Counter;
		ColorsUChar[3] = 255;

		voxels[tid] = Colors;*/

		float4* VoxelsFloat4 = (float4*)voxels;
		VoxelsFloat4[tid] = make_float4(r/Counter, g/Counter, b/Counter, 0);
	}
}


#endif // VOLUMEMANAGERKERNEL_H__