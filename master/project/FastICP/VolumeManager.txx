#include <algorithm>
#include <iterator>

#include "VolumeManager.h"
#include "defs.h"
#include "ritkDebugManager.h"


//----------------------------------------------------------------------------
extern "C"
void CUDAAddPointsToVolume(float* points, float* voxels, float* origin, unsigned long numPts, unsigned int dimSize, unsigned int spacing);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
template<unsigned int DimSize, unsigned int Spacing>
VolumeManager<DimSize, Spacing>::VolumeManager()
{
	// Init container for origin
	m_Origin = DatasetContainer::New();
	DatasetContainer::SizeType OriginSize;
	OriginSize.SetElement(0, 3);
	m_Origin->SetContainerSize(OriginSize);
	m_Origin->Reserve(OriginSize[0]);
	SetOrigin(-2560, -2560, -2560); // Default position of origin

	// Init container for all voxels
	m_Voxels = VoxelContainer::New();
	VoxelContainer::SizeType VoxelsSize;
	VoxelsSize.SetElement(0, DimSize*DimSize*DimSize*4);
	m_Voxels->SetContainerSize(VoxelsSize);
	m_Voxels->Reserve(VoxelsSize[0]);
	ResetVolume(); // Reset all voxels in the volume

	// Allocate memory for occupied points storage
	m_OccupiedPoints = new float[DimSize*DimSize*DimSize*ICP_DATA_DIM*3];
}


//----------------------------------------------------------------------------
template<unsigned int DimSize, unsigned int Spacing>
VolumeManager<DimSize, Spacing>::~VolumeManager()
{
	delete [] m_OccupiedPoints;
}


//----------------------------------------------------------------------------
template<unsigned int DimSize, unsigned int Spacing>
void VolumeManager<DimSize, Spacing>::SetOrigin(float x, float y, float z)
{
	m_Origin->GetBufferPointer()[0] = x;
	m_Origin->GetBufferPointer()[1] = y;
	m_Origin->GetBufferPointer()[2] = z;

	m_Origin->SynchronizeDevice();
}


//----------------------------------------------------------------------------
template<unsigned int DimSize, unsigned int Spacing>
void VolumeManager<DimSize, Spacing>::ResetVolume()
{
	// Fill all voxels with value -1 (unoccupied)
	std::fill_n(m_Voxels->GetBufferPointer(), DimSize*DimSize*DimSize*4, -1.f);

	// Write to device
	m_Voxels->SynchronizeDevice();
}


//----------------------------------------------------------------------------
template<unsigned int DimSize, unsigned int Spacing>
void VolumeManager<DimSize, Spacing>::AddPoints(DatasetContainer::Pointer Points)
{
	CUDAAddPointsToVolume(
		Points->GetCudaMemoryPointer(),
		m_Voxels->GetCudaMemoryPointer(),
		m_Origin->GetCudaMemoryPointer(),
		Points->Size()/ICP_DATA_DIM,
		DimSize,
		Spacing
		);
}


//----------------------------------------------------------------------------
template<unsigned int DimSize, unsigned int Spacing>
float* VolumeManager<DimSize, Spacing>::GetOccupiedPoints(unsigned long& NumPoints)
{
	m_Voxels->SynchronizeHost();
	float* Voxels = m_Voxels->GetBufferPointer();

	NumPoints = 0;
	for (int i = 0; i < DimSize*DimSize*DimSize; ++i)
	{
		if (Voxels[i*4+0] != -1.f)
		{
			int Tmp = i;
			int zUnits = Tmp % DimSize;
			Tmp /= DimSize;
			int yUnits = Tmp % DimSize;
			Tmp /= DimSize;
			int xUnits = Tmp % DimSize;

			float x = m_Origin->GetBufferPointer()[0] + xUnits*Spacing;
			float y = m_Origin->GetBufferPointer()[1] + yUnits*Spacing;
			float z = m_Origin->GetBufferPointer()[2] + zUnits*Spacing;

			//// all color values are stored within a 4-byte-block as uchars
			//uchar* VoxelsUChar = (uchar*)Voxels;
			//uchar r = VoxelsUChar[i*sizeof(float)+0];
			//uchar g = VoxelsUChar[i*sizeof(float)+1];
			//uchar b = VoxelsUChar[i*sizeof(float)+2];

			//m_OccupiedPoints[NumPoints*ICP_DATA_DIM+0] = x;
			//m_OccupiedPoints[NumPoints*ICP_DATA_DIM+1] = y;
			//m_OccupiedPoints[NumPoints*ICP_DATA_DIM+2] = z;
			//m_OccupiedPoints[NumPoints*ICP_DATA_DIM+3] = r;
			//m_OccupiedPoints[NumPoints*ICP_DATA_DIM+4] = g;
			//m_OccupiedPoints[NumPoints*ICP_DATA_DIM+5] = b;

			float r = Voxels[i*4+0];
			float g = Voxels[i*4+1];
			float b = Voxels[i*4+2];

			m_OccupiedPoints[NumPoints*ICP_DATA_DIM+0] = x;
			m_OccupiedPoints[NumPoints*ICP_DATA_DIM+1] = y;
			m_OccupiedPoints[NumPoints*ICP_DATA_DIM+2] = z;
			m_OccupiedPoints[NumPoints*ICP_DATA_DIM+3] = r;
			m_OccupiedPoints[NumPoints*ICP_DATA_DIM+4] = g;
			m_OccupiedPoints[NumPoints*ICP_DATA_DIM+5] = b;

			++NumPoints;
		}
	}

	return m_OccupiedPoints;
}