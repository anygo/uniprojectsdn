#ifndef VOLUMEMANAGERKERNEL_H__
#define VOLUMEMANAGERKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "defs.h"


//----------------------------------------------------------------------------
__global__ void kernelAddPointsToVolumePointToVoxel(float* points, unsigned char* voxels, float* config, unsigned long numPts)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= numPts ) return;

	// Get config
	float3* ConfigFloat3 = (float3*)config;
	const float3 Origin = ConfigFloat3[0];
	const float3 Spacing = ConfigFloat3[1];
	const float3 Dim = ConfigFloat3[2];

	// Compute total number of voxels
	const int NumVoxels = Dim.x*Dim.y*Dim.z;

	// Get this thread's point world coordinates
	const float3 Coords = *((float3*)(points+tid*ICP_DATA_DIM));
	const float3 Color  = *((float3*)(points+(tid*ICP_DATA_DIM+3)));

	// Compute volume coordinates
	const int x = (Coords.x-Origin.x)/Spacing.x;
	const int y = (Coords.y-Origin.y)/Spacing.y;
	const int z = (Coords.z-Origin.z)/Spacing.z;

	// Check if voxel is inside specified volume
	if (x < 0 || x >= Dim.x || y < 0 || y >= Dim.y || z < 0 || z >= Dim.z)
		return;

	// Create color uchar4
	const uchar4 ColorUChar4 = make_uchar4(Color.x, Color.y, Color.z, 255);

	// Compute voxel index of linearized volume
	int VoxelIdx = z*Dim.x*Dim.y + y*Dim.z + x;

	// Write to global memory
	if (VoxelIdx >= 0 && VoxelIdx < NumVoxels)
	{

#if 1
		atomicExch((unsigned int*)(voxels+(VoxelIdx*sizeof(uchar4))), *((unsigned int*)&ColorUChar4));
#else
		// Cast voxels to uchar4*
		uchar4* VoxelsUChar4 = (uchar4*)voxels;
		VoxelsUChar4[VoxelIdx] = ColorUChar4;
#endif

	}
}


#endif // VOLUMEMANAGERKERNEL_H__