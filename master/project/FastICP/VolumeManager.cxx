#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>

#include "VolumeManager.h"
#include "defs.h"
#include "ritkDebugManager.h"


//----------------------------------------------------------------------------
extern "C"
void CUDAAddPointsToVolumeVoxelToPoint(float* points, uchar* voxels, float* config, unsigned long numPts, int numVoxels);

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
}


//----------------------------------------------------------------------------
VolumeManager::~VolumeManager()
{
}


//----------------------------------------------------------------------------
void
VolumeManager::SetRegions(RegionType region)
{
	SetLargestPossibleRegion(region);
	SetBufferedRegion(region);
	SetRequestedRegion(region);
}


//----------------------------------------------------------------------------
void
VolumeManager::SetRegions(SizeType size)
{
	RegionType region; 
	region.SetSize(size);
	SetRegions(region);
}


//----------------------------------------------------------------------------
void
VolumeManager::SetLargestPossibleRegion(const RegionType &Region)
{
	// Update the superclass
	Superclass::SetLargestPossibleRegion(Region);
}


//----------------------------------------------------------------------------
void
VolumeManager::SetRequestedRegion(const RegionType &Region)
{
	// Update the superclass
	Superclass::SetRequestedRegion(Region);
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
	VoxelsSize[0] = RegionSize[0]*RegionSize[1]*RegionSize[2]*4;
	m_Voxels->SetContainerSize(VoxelsSize);
	m_Voxels->Reserve(VoxelsSize[0]);

	// Reset all voxels in the volume
	ResetVolume();

	// Now let's use our CUDARegularMemoryImportImageContainer instead of the itk::Image's standard ImportImageContainer
	this->SetPixelContainer(reinterpret_cast<Superclass::PixelContainer*>(m_Voxels.GetPointer()));
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
VolumeManager::SetSpacing(const double Spacing[ImageDimension])
{
	SpacingType s(Spacing);
	this->SetSpacing(s);
}


//----------------------------------------------------------------------------
void
VolumeManager::SetSpacing(const float Spacing[ImageDimension])
{
	itk::Vector<float, ImageDimension> sf(Spacing);
	SpacingType s;
	s.CastFrom(sf);
	this->SetSpacing(s);
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
VolumeManager::SetOrigin(const double Origin[ImageDimension])
{
	PointType p(Origin);
	this->SetOrigin(p);
}


//----------------------------------------------------------------------------
void
VolumeManager::SetOrigin(const float Origin[ImageDimension])
{
	itk::Point<float, ImageDimension> of(Origin);
	PointType p;
	p.CastFrom(of);
	this->SetOrigin(p);
}


//----------------------------------------------------------------------------
void
VolumeManager::PrintSelf(std::ostream& os, itk::Indent indent) const
{
	Superclass::PrintSelf(os,indent);
	os << indent << "VolumeManager" << std::endl;
}


//----------------------------------------------------------------------------
void
VolumeManager::ResetVolume()
{
	// Fill all voxels with value 0 (unoccupied)
	const size_t VoxelArraySize = m_Voxels->GetContainerSize()[0];
	std::fill_n(stdext::checked_array_iterator<uchar*>(m_Voxels->GetBufferPointer(), VoxelArraySize), VoxelArraySize, 0);

	// Write to device
	m_Voxels->SynchronizeDevice();
}


//----------------------------------------------------------------------------
void
VolumeManager::AddPoints(DatasetContainer::Pointer Points)
{
	const int NumPts = Points->Size()/ICP_DATA_DIM;
	const int numVoxels = m_Voxels->GetContainerSize()[0]/4;

#if 0
	// Each voxel finds all corresponding points
	const int NumPtsAtATime = 4096;

	int CurOffset = 0;
	for (int PtsLeft = NumPts; PtsLeft > 0; PtsLeft -= NumPtsAtATime)
	{
		// Synchronizing is required, since otherwise the graphics card driver will crash
		cudaThreadSynchronize();

		// Print number of remaining points (because execution of this method takes a while)
		std::cout << "\r" << PtsLeft << " points remaining...      ";

		int CurNumPts = PtsLeft < NumPtsAtATime ? PtsLeft : NumPtsAtATime;

		CUDAAddPointsToVolumeVoxelToPoint(
			Points->GetCudaMemoryPointer()+CurOffset,
			m_Voxels->GetCudaMemoryPointer(),
			m_Config->GetCudaMemoryPointer(),
			CurNumPts,
			numVoxels
			);

		CurOffset += NumPtsAtATime*ICP_DATA_DIM;
	}

	// Clear line
	std::cout << "\r"; std::fill_n(std::ostream_iterator<char>(std::cout), 60, ' '); std::cout << "\r";
#else
	// Each point finds its corresponding voxel
	CUDAAddPointsToVolumePointToVoxel(
		Points->GetCudaMemoryPointer(),
		m_Voxels->GetCudaMemoryPointer(),
		m_Config->GetCudaMemoryPointer(),
		NumPts
		);
#endif
}


//----------------------------------------------------------------------------
float*
VolumeManager::GetOccupiedPoints(unsigned long& NumPoints)
{
	// Synchronize volume data to host
	m_Voxels->SynchronizeHost();

	// For convenience
	uchar* Voxels = m_Voxels->GetBufferPointer();

	// Clear vector
	m_OccupiedPoints.clear();

	// Get size of each dimension
	SizeType Size = this->GetBufferedRegion().GetSize();

	// Init counting variable
	NumPoints = 0;

	// Iterate over all voxels
	for (int i = 0; i < Size[0]*Size[1]*Size[2]; ++i)
	{
		if (Voxels[i*4+0] != 0)
		{
			// Compute spatial coordinates of the current voxel
			int Tmp = i;
			int zUnits = Tmp % static_cast<int>(Size[2]);
			Tmp /= static_cast<int>(Size[2]);
			int yUnits = Tmp % static_cast<int>(Size[1]);
			Tmp /= static_cast<int>(Size[1]);
			int xUnits = Tmp % static_cast<int>(Size[0]);

			float x = m_Origin[0] + xUnits*m_Spacing[0];
			float y = m_Origin[1] + yUnits*m_Spacing[1];
			float z = m_Origin[2] + zUnits*m_Spacing[2];

			float r = Voxels[i*4+0];
			float g = Voxels[i*4+1];
			float b = Voxels[i*4+2];

			// Add coordinates and color to vector
			m_OccupiedPoints.push_back(make_float3(x, y, z));
			m_OccupiedPoints.push_back(make_float3(r, g, b));

			// Increment counter for valid points
			++NumPoints;
		}
	}

	// This might not be C++ standard? However, it works for MS VC++ 2010 with SP1 and seems very convenient
	return (float*)m_OccupiedPoints.data();
}