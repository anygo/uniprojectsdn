#ifndef VOLUMEMANAGER_H__
#define	VOLUMEMANAGER_H__

#include "ritkCudaRegularMemoryImportImageContainer.h"


/**	@class		VolumeManager
 *	@brief		Class for building and updating a volume
 *	@author		Dominik Neumann
 *
 *	@details
 *	This class handles voxels of a quantized 3D volume and allows updating itself using point clouds
 */
template<unsigned int DimSize, unsigned int Spacing>
class VolumeManager
{
	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerF VoxelContainer;
	//@}

public:
	/// Constructor 
	VolumeManager();

	/// Destructor
	~VolumeManager();

	/// Set position of origin in world coordinates
	inline void SetOrigin(float x, float y, float z);

	/// Incorporate a new point cloud into volume (assumes linearized 6D data (XYZRGB)
	void AddPoints(DatasetContainer::Pointer Points);

	/// Returns all 'occupied' points in the volume as a float pointer
	float* GetOccupiedPoints(unsigned long& NumPoints);

	/// Reset the volume
	void ResetVolume();
	
protected:
	/// The origin of the cube / volume
	DatasetContainer::Pointer m_Origin;

	/// Container that holds all voxels
	VoxelContainer::Pointer m_Voxels;

	/// Reserverd memory for writing all occupied points to, once GetOccupiedPoints is called
	float* m_OccupiedPoints;

private:
	/// Purposely not implemented
	VolumeManager(VolumeManager&);

	/// Purposely not implemented
	void operator=(const VolumeManager&); 
};


#endif // VOLUMEMANAGER_H__