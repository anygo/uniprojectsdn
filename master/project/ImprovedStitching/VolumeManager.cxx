#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>

#include "VolumeManager.h"
#include "defs.h"
#include "ritkDebugManager.h"
#include "itkImageRegionConstIterator.h"


//----------------------------------------------------------------------------
extern "C"
void CUDAAddPointsToVolumePointToVoxel(float* points, uchar* voxels, float* config, unsigned long numPts);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
VolumeManager::VolumeManager()
{
	// Init container for config data (origin, spacing, size)
	m_Config = DatasetContainer::New();
	DatasetContainer::SizeType ConfigSize;
	ConfigSize.SetElement(0, 9);
	m_Config->SetContainerSize(ConfigSize);
	m_Config->Reserve(ConfigSize[0]);

	m_AddPointsCalls = 0;
}


//----------------------------------------------------------------------------
VolumeManager::~VolumeManager()
{
}


//----------------------------------------------------------------------------
void
VolumeManager::SetBufferedRegion(const RegionType &Region)
{
	// Update the superclass
	Superclass::SetBufferedRegion(Region);

	m_Config->GetBufferPointer()[6] = Region.GetSize()[0]; // Dimension extent is at pos 6,7,8
	m_Config->GetBufferPointer()[7] = Region.GetSize()[1];
	m_Config->GetBufferPointer()[8] = Region.GetSize()[2];

	m_Config->SynchronizeDevice();
}


//----------------------------------------------------------------------------
void
VolumeManager::Allocate()
{
	// Init container for all voxels
	m_Voxels = VoxelContainer::New();

	SizeType RegionSize = this->GetLargestPossibleRegion().GetSize();
	VoxelContainer::SizeType VoxelsSize;	
	VoxelsSize[0] = RegionSize[0]*RegionSize[1]*RegionSize[2];
	m_Voxels->SetContainerSize(VoxelsSize);
	m_Voxels->Reserve(VoxelsSize[0]);

	// Reset all voxels in the volume
	ResetVolume();

	// Now let's use our CUDARegularMemoryImportImageContainer instead of the itk::Image's standard ImportImageContainer
	this->SetPixelContainer(m_Voxels);
}


//----------------------------------------------------------------------------
void
VolumeManager::SetSpacing(const SpacingType &Spacing)
{
	// Update the superclass
	Superclass::SetSpacing(Spacing);

	m_Config->GetBufferPointer()[3] = Spacing[0]; // Spacing is at pos 3,4,5
	m_Config->GetBufferPointer()[4] = Spacing[1];
	m_Config->GetBufferPointer()[5] = Spacing[2];

	m_Config->SynchronizeDevice();
}


//----------------------------------------------------------------------------
void
VolumeManager::SetOrigin(const PointType &Origin)
{
	// Update the superclass
	Superclass::SetOrigin(Origin);

	m_Config->GetBufferPointer()[0] = Origin[0]; // Spacing is at pos 3,4,5
	m_Config->GetBufferPointer()[1] = Origin[1];
	m_Config->GetBufferPointer()[2] = Origin[2];

	m_Config->SynchronizeDevice();
}


//----------------------------------------------------------------------------
void
VolumeManager::PrintSelf(std::ostream& os, itk::Indent indent) const
{
	Superclass::PrintSelf(os, indent);
	os << indent << "VolumeManager" << std::endl;
}


//----------------------------------------------------------------------------
void
VolumeManager::ResetVolume()
{
	LOG_DEB("ResetVolume() after " << m_AddPointsCalls << " calls to AddPoints()");
	m_AddPointsCalls = 0;

	// Fill all voxels with value 0 (unoccupied)
	const size_t VoxelArraySize = m_Voxels->GetContainerSize()[0];

	PixelType ZeroPixel;
	ZeroPixel.Fill(0);

	std::fill_n(stdext::checked_array_iterator<PixelType*>(m_Voxels->GetBufferPointer(), VoxelArraySize), VoxelArraySize, ZeroPixel);

	// Write to device
	m_Voxels->SynchronizeDevice();
}


//----------------------------------------------------------------------------
void
VolumeManager::AddPoints(DatasetContainer::Pointer Points)
{
	++m_AddPointsCalls;

	const int NumPts = Points->Size()/ICP_DATA_DIM;
	const int numVoxels = m_Voxels->GetContainerSize()[0];

	// Each point finds its corresponding voxel
	CUDAAddPointsToVolumePointToVoxel(
		Points->GetCudaMemoryPointer(),
		reinterpret_cast<uchar*>(m_Voxels->GetCudaMemoryPointer()),
		m_Config->GetCudaMemoryPointer(),
		NumPts
		);	
}


//----------------------------------------------------------------------------
float*
VolumeManager::GetOccupiedPoints(unsigned long& NumPoints)
{
	// Synchronize volume data to host
	m_Voxels->SynchronizeHost();

	// Clear vector
	m_OccupiedPoints.clear();

	// Init counting variable
	NumPoints = 0;

	typedef itk::ImageRegionConstIterator<VolumeManager> IteratorType;
	IteratorType ItVoxels(this, this->GetBufferedRegion());

	for (ItVoxels.GoToBegin(); !ItVoxels.IsAtEnd(); ++ItVoxels)
	{
		// Get current voxel
		PixelType curVoxel = ItVoxels.Value();
		if (curVoxel.GetAlpha() > 0) // Only visualize voxels with alpha value greater than 0
		{
			PointType Coords;

			// Compute world coordinate of that voxel
			TransformIndexToPhysicalPoint(ItVoxels.GetIndex(), Coords);

			// Add coordinates and color to vector
			m_OccupiedPoints.push_back(make_float3(Coords[0], Coords[1], Coords[2]));
			m_OccupiedPoints.push_back(make_float3(curVoxel[0], curVoxel[1], curVoxel[2]));

			// Increment counter for valid points
			++NumPoints;
		}
	}

	// Not sure, if this will work with every compiler
	return (float*)(&m_OccupiedPoints[0]);
}