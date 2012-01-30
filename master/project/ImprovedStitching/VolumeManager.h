#ifndef VOLUMEMANAGER_H__
#define	VOLUMEMANAGER_H__

#include "itkImage.h"
#include "itkRGBAPixel.h"

#include "ritkCudaRegularMemoryImportImageContainer.h"
#include "ritkCuda3DArrayImportImageContainer.h"
#include "RBC.h"


/**	@class		VolumeManager
 *	@brief		Class for building and updating a volume
 *	@author		Dominik Neumann
 *
 *	@details
 *	This class handles voxels of a quantized 3D volume and allows updating itself using point clouds
 */
class VolumeManager : public itk::Image<itk::RGBAPixel<uchar>,3>
{
public:
	/** @name Standard typedefs */
	//@{
	typedef VolumeManager								Self;
	typedef itk::Image<itk::RGBAPixel<uchar>,3>			Superclass;
	typedef itk::SmartPointer<Self>						Pointer;
	typedef itk::SmartPointer<const Self>				ConstPointer;
	typedef itk::WeakPointer<const Self>				ConstWeakPointer;
	//@}

	/**	@name itk::Image meta information typedefs */
	//@{
	typedef Superclass::IndexType						IndexType;
	typedef Superclass::SizeType						SizeType;
	typedef Superclass::RegionType						RegionType;
	typedef Superclass::SpacingType						SpacingType;
	typedef Superclass::DirectionType					DirectionType;
	typedef Superclass::PointType						PointType;
	typedef Superclass::PixelType						PixelType;
	typedef Superclass::InternalPixelType				InternalPixelType;
	//@}

	/// Object creation
	itkNewMacro(Self);

	/// RTTI
	itkTypeMacro(VolumeManager, Superclass);

	/// Image dimension
	itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

	/// Override required, since the size of the volume needs to be copied to GPU
	virtual void SetBufferedRegion(const RegionType &region);

	/// Override required, since spacing needs to be copied to GPU
	virtual void SetSpacing(const SpacingType &Spacing);

	/// Override required, since origin needs to be copied to GPU
	virtual void SetOrigin(const PointType &Origin);

	/// Allocate the volume
	virtual void Allocate();


	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF								DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainer<ulong, itk::RGBAPixel<uchar> >	VoxelContainer;
	//@}
	

	/// Incorporate a new point cloud into volume (assumes linearized 6D data (XYZRGB)
	virtual void AddPoints(DatasetContainer::Pointer Points);

	/// Returns all 'occupied' points in the volume as a float pointer
	virtual float* GetOccupiedPoints(unsigned long& NumPoints);

	/// Reset the volume
	virtual void ResetVolume();

	/// Synchronize the voxels to host
	virtual void SynchronizeDataToHost() { m_Voxels->SynchronizeHost(); }
	
protected:
	/// Constructor 
	VolumeManager();

	/// Destructor
	virtual ~VolumeManager();

	/// Print the object on a stream
	virtual void PrintSelf(std::ostream& os, itk::Indent indent) const;

	/// Container that holds all voxels
	VoxelContainer::Pointer m_Voxels;

	/// Vector for writing all occupied points to, once GetOccupiedPoints is called (used for conversion to point cloud)
	std::vector<float3> m_OccupiedPoints;

	/// Configuration container for spacing, dimension extent and origin (since we need it on the GPU as well)
	DatasetContainer::Pointer m_Config;

	int m_AddPointsCalls;

private:
	/// Purposely not implemented
	VolumeManager(VolumeManager&);

	/// Purposely not implemented
	void operator=(const VolumeManager&); 
};


#endif // VOLUMEMANAGER_H__