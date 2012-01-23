#ifndef VOLUMEMANAGER_H__
#define	VOLUMEMANAGER_H__

#include "itkImage.h"
#include "itkRGBAPixel.h"

#include "ritkCudaRegularMemoryImportImageContainer.h"
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
	typedef Superclass::InternalPixelType				InternalPixelType;
	//@}

	/// Object creation
	itkNewMacro(Self);

	/// RTTI
	itkTypeMacro(VolumeManager, Superclass);

	/// Image dimension
	itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

	/** @name Convenience methods to set the LargestPossibleRegion, BufferedRegion and RequestedRegion. Allocate must still be called.*/
	//@{
	virtual void SetRegions(RegionType region);
	virtual void SetRegions(SizeType size);
	//@}

	/**	@name LargestPossibleRegion, BufferedRegion, RequestedRegion.*/
	//@{
	virtual void SetLargestPossibleRegion(const RegionType &region);
	virtual void SetRequestedRegion(const RegionType &region);
	virtual void SetBufferedRegion(const RegionType &region);
	//@}

	/** @name Set the spacing.*/
	//@{
	virtual void SetSpacing(const SpacingType &Spacing);
	virtual void SetSpacing(const double Spacing[ImageDimension]);
	virtual void SetSpacing(const float Spacing[ImageDimension]);
	//@}

	/** @name Set the origin.*/
	//@{
	virtual void SetOrigin(const PointType &Origin);
	virtual void SetOrigin(const double Origin[ImageDimension]);
	virtual void SetOrigin(const float Origin[ImageDimension]);
	//@}

	/// Allocate the volume
	virtual void Allocate();


	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF				DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainer<ulong, uchar>	VoxelContainer;
	//@}
	

	/// Incorporate a new point cloud into volume (assumes linearized 6D data (XYZRGB)
	virtual void AddPoints(DatasetContainer::Pointer Points);

	/// Returns all 'occupied' points in the volume as a float pointer
	virtual float* GetOccupiedPoints(unsigned long& NumPoints);

	/// Reset the volume
	virtual void ResetVolume();
	
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

private:
	/// Purposely not implemented
	VolumeManager(VolumeManager&);

	/// Purposely not implemented
	void operator=(const VolumeManager&); 
};


#endif // VOLUMEMANAGER_H__