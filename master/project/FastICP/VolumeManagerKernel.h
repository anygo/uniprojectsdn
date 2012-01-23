#ifndef VOLUMEMANAGERKERNEL_H__
#define VOLUMEMANAGERKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "defs.h"


//----------------------------------------------------------------------------
__global__ void kernelAddPointsToVolumeVoxelToPoint(float* points, unsigned char* voxels, float* config, unsigned long numPts)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	// Get config
	float3* ConfigFloat3 = (float3*)config;
	float3 Origin = ConfigFloat3[0];
	float3 Spacing = ConfigFloat3[1];
	float3 Dim = ConfigFloat3[2];

	if ( tid >= Dim.x*Dim.y*Dim.z ) return;

	// Compute x-, y- and z-boundaries of this thread's voxel
	int Tmp = tid;
	int zUnits = Tmp % (int)Dim.z;
	Tmp /= (int)Dim.z;
	int yUnits = Tmp % (int)Dim.y;
	Tmp /= (int)Dim.y;
	int xUnits = Tmp % (int)Dim.x;

	float xMin = Origin.x + (xUnits  )*Spacing.x;
	float xMax = Origin.x + (xUnits+1)*Spacing.x;

	float yMin = Origin.y + (yUnits  )*Spacing.y;
	float yMax = Origin.y + (yUnits+1)*Spacing.y;

	float zMin = Origin.z + (zUnits  )*Spacing.z;
	float zMax = Origin.z + (zUnits+1)*Spacing.z;

	// For each point, determine if it lies inside the voxel handled by this thread
	float r = 0, g = 0, b = 0;
	float x, y, z;
	float Counter = 0;
	for (int i = 0; i < numPts; ++i)
	{
		// Get the i'th point's spatial coordinates
		const float3 Coords = *((float3*)(points+tid*ICP_DATA_DIM));
		x = Coords.x;
		y = Coords.y;
		z = Coords.z;

		if (x >= xMin && x < xMax && y >= yMin && y < yMax && z >= zMin && z < zMax)
		{
			// Accumulate color
			const float3 Color = *((float3*)(points+(tid*ICP_DATA_DIM+3)));
			r += Color.x;
			g += Color.y;
			b += Color.z;

			++Counter;
		}
	}

	// If at least one point is inside this voxel, store the average color
	if (Counter > 0)
	{
		// Store colors in uchar4*-casted voxels-pointer
		uchar4* VoxelsFloat4 = (uchar4*)voxels;
		VoxelsFloat4[tid] = make_uchar4(r/Counter, g/Counter, b/Counter, 0);
	}
}


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

	// Cast voxels to uchar4*
	uchar4* VoxelsUChar4 = (uchar4*)voxels;

#if 0
	// Blurring (also use neighboring voxels)	
	#pragma unroll
	for (int xBlur = -1; xBlur < 1; ++xBlur)
	{
		#pragma unroll
		for (int yBlur = -1; yBlur < 1; ++yBlur)
		{
			#pragma unroll
			for (int zBlur = -1; zBlur < 1; ++zBlur)
			{
				// Compute voxel index of linearized volume
				int VoxelIdx = (x+xBlur)*Dim.y*Dim.z + (y+yBlur)*Dim.z + (z+zBlur);

				// Write to global memory
				if (VoxelIdx >= 0 && VoxelIdx < NumVoxels)
				{
					// Get previous color of that voxel
					uchar4 PrevColor = VoxelsUChar4[VoxelIdx];

					// If appropriate, compute mean of previous and current color, otherwise use current color
					if (*((float*)&PrevColor) > 0)
						VoxelsUChar4[VoxelIdx] = make_uchar4((PrevColor.x+ColorUChar4.x)/2, (PrevColor.y+ColorUChar4.y)/2, (PrevColor.z+ColorUChar4.z)/2, 0);
					else
						VoxelsUChar4[VoxelIdx] = ColorUChar4;
				}		
			}
		}
	}
#else
	// Compute voxel index of linearized volume
	int VoxelIdx = x*Dim.y*Dim.z + y*Dim.z + z;

	// Write to global memory
	if (VoxelIdx >= 0 && VoxelIdx < NumVoxels)
	{
		// Get previous color of that voxel
		uchar4 PrevColor = VoxelsUChar4[VoxelIdx];

		// If appropriate, compute mean of previous and current color, otherwise use current color
		if (*((float*)&PrevColor) > 0)
			VoxelsUChar4[VoxelIdx] = make_uchar4((PrevColor.x+ColorUChar4.x)/2, (PrevColor.y+ColorUChar4.y)/2, (PrevColor.z+ColorUChar4.z)/2, 0);
		else
			VoxelsUChar4[VoxelIdx] = ColorUChar4;
	}
#endif
}


#endif // VOLUMEMANAGERKERNEL_H__