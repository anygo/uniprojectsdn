#ifndef KINECTDATAMANAGER_H__
#define KINECTDATAMANAGER_H__

#include "ritkCudaRegularMemoryImportImageContainer.h"
#include "ritkRImage.h"
#include "ritkRImageActorPipeline.h"


/**	@class		DataGenerator
 *	@brief		Class that generates synthetic data
 *	@author		Dominik Neumann
 *
 *	@details
 *	This is a class that generates two randomly distributed RGB-colored 3-D point sets that are both identical, except
 *	that the second point set can be transformed and made noisy.
 */
class KinectDataManager : public itk::Object
{
public:
	/**	@name Standard ITK typedefs */
	//@{
	typedef KinectDataManager								Self;
	typedef itk::Object										Superclass;
	typedef itk::SmartPointer<Self>							Pointer;
	typedef itk::SmartPointer<const Self>					ConstPointer;
	//@}

	/// Object creation
	itkNewMacro(Self);

	/// RTTI
	itkTypeMacro(OpenGLSkeletonEntity, itk::Object);

	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerU IndicesContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerF MatrixContainer;
	//@}

	/// Import a frame
	virtual void ImportKinectData(ritk::NewFrameEvent::RImageConstPointer Data);

	/// Swap data
	virtual void SwapPointsContainer(KinectDataManager* Other);

	/// Set number of points per dataset
	virtual void SetNumberOfLandmarks(unsigned long NumLandmarks);

	/// Set clip percentage for landmark extraction
	virtual void SetClipPercentage(float ClipPercentage);

	/// Extract landmarks from current set of points
	virtual void ExtractLandmarks();

	/// Apply transformation to all points
	virtual void TransformPts(MatrixContainer::Pointer Mat);

	/// Returns pointer to the dataset container holding the fixed points
	virtual inline DatasetContainer::Pointer GetPtsContainer() const { return m_Pts; }

	/// Returns pointer to the dataset container holding the fixed points
	virtual inline DatasetContainer::Pointer GetLandmarkContainer() const { return m_Landmarks; }
	
protected:
	/// Constructor 
	KinectDataManager();

	/// Destructor
	virtual ~KinectDataManager();

	/// Update the indices for the landmark extraction
	virtual void UpdateLandmarkIndices();

	/// Pointer to set of all points
	DatasetContainer::Pointer m_Pts;

	/// Pointer to set of landmarks
	DatasetContainer::Pointer m_Landmarks;

	/// Pointer to set of indices used during landmark extraction
	IndicesContainer::Pointer m_LandmarkIndices;

	/// Stores number of points
	unsigned long m_NumLandmarks;

	/// Stores clip percentage
	float m_ClipPercentage;

	/// Memory used during 2-D/3-D transformation
	cudaArray* m_InputImgArr;

private:
	/// Purposely not implemented
	KinectDataManager(KinectDataManager&);

	/// Purposely not implemented
	void operator=(const KinectDataManager&); 
};


#endif // KINECTDATAMANAGER_H__