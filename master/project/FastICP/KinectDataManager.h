#ifndef KINECTDATAMANAGER_H__
#define KINECTDATAMANAGER_H__

#include "ritkCudaRegularMemoryImportImageContainer.h"
#include "ritkRImage.h"
#include "ritkRImageActorPipeline.h"

#include "vtkSmartPointer.h"


/**	@class		DataGenerator
 *	@brief		Class that generates synthetic data
 *	@author		Dominik Neumann
 *
 *	@details
 *	This is a class that generates two randomly distributed RGB-colored 3-D point sets that are both identical, except
 *	that the second point set can be transformed and made noisy.
 */
class KinectDataManager
{
	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerU IndicesContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerF MatrixContainer;
	//@}

public:
	/// Constructor 
	KinectDataManager(unsigned long NumLandmarks = 2048, float ClipPercentage = 0.1f);

	/// Destructor
	~KinectDataManager();

	/// Import a frame
	void ImportKinectData(ritk::NewFrameEvent::RImageConstPointer Data);

	/// Set number of points per dataset
	void SetNumberOfLandmarks(unsigned long NumLandmarks);

	/// Set clip percentage for landmark extraction
	void SetClipPercentage(float ClipPercentage);

	/// Apply transformation to all points
	void TransformPts(MatrixContainer::Pointer Mat);

	/// Returns pointer to the dataset container holding the fixed points
	inline DatasetContainer::Pointer GetPtsContainer() const { return m_Pts; }

	/// Returns pointer to the dataset container holding the fixed points
	inline DatasetContainer::Pointer GetLandmarkContainer() const { return m_Landmarks; }
	
protected:
	/// Update the indices for the landmark extraction
	void UpdateLandmarkIndices();

	/// Pointer to set of all points
	DatasetContainer::Pointer m_Pts;

	/// Pointer to set of landmarks
	DatasetContainer::Pointer m_Landmarks;

	/// Pointer to set of indices used during landmark extraction
	IndicesContainer::Pointer m_LandmarkIndices;

	/// Stores number of points
	unsigned long m_NumLandmarks;
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