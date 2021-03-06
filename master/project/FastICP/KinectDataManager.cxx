#include "KinectDataManager.h"
#include "ritkRGBRImage.h"
#include "defs.h"

#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"
#include "itkImageRegionConstIterator.h"


//----------------------------------------------------------------------------
extern "C"
void CUDATransformPoints3D(float* points, float* m, unsigned int numPts, unsigned int dim);

extern "C"
void CUDARangeToWorld(float* pointsOut, const cudaArray* inputImageArray);

extern "C"
void CUDAExtractLandmarks(float* landmarksOut, float* pointsIn, unsigned long* indices, unsigned long numLandmarks);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
KinectDataManager::KinectDataManager()
{
	// Default number of landmarks
	const int NumLandmarks = 2048;

	// Default clip percentage
	m_ClipPercentage = 0.1;

	// Init container for set of all points (including landmarks)
	m_Pts = DatasetContainer::New();
	DatasetContainer::SizeType DataSize;
	DataSize.SetElement(0, ICP_DATA_DIM * KINECT_IMAGE_WIDTH * KINECT_IMAGE_HEIGHT);
	m_Pts->SetContainerSize(DataSize);
	m_Pts->Reserve(DataSize[0]);

	// Init container for set of landmarks
	m_Landmarks = DatasetContainer::New();
	DatasetContainer::SizeType LandmarksSize;
	LandmarksSize.SetElement(0, ICP_DATA_DIM * NumLandmarks);
	m_Landmarks->SetContainerSize(LandmarksSize);
	m_Landmarks->Reserve(LandmarksSize[0]);

	// Init container for set of landmark indices
	m_LandmarkIndices = IndicesContainer::New();
	IndicesContainer::SizeType IndicesSize;
	IndicesSize.SetElement(0, NumLandmarks);
	m_LandmarkIndices->SetContainerSize(IndicesSize);
	m_LandmarkIndices->Reserve(IndicesSize[0]);

	// Init number of points (including landmark indices generation) and clip percentage
	m_NumLandmarks = 0;
	SetNumberOfLandmarks(NumLandmarks);

	// Allocate memory on GPU for 2-D/3-D conversion
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	ritkCudaSafeCall(cudaMallocArray(&m_InputImgArr, &ChannelDesc, KINECT_IMAGE_WIDTH, KINECT_IMAGE_HEIGHT));
}


//----------------------------------------------------------------------------
KinectDataManager::~KinectDataManager()
{
	ritkCudaSafeCall(cudaFreeArray(m_InputImgArr));
}


//----------------------------------------------------------------------------
void
KinectDataManager::ImportKinectData(ritk::NewFrameEvent::RImageConstPointer Data)
{
	// Set RGB information
	float* PtsPtr = m_Pts->GetBufferPointer();

	ritk::RGBRImageUCF2::ConstPointer CurrentFrameRGBP = dynamic_cast<const ritk::RGBRImageUCF2*>((const ritk::RImageF2*) Data);
	typedef itk::ImageRegionConstIterator<ritk::RGBRImageUCF2::RGBImageType> IteratorType;
	IteratorType it(CurrentFrameRGBP->GetRGBImage(), CurrentFrameRGBP->GetRGBImage()->GetRequestedRegion());
	it.GoToBegin();
	for (int pt = 0; pt < KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT; ++pt, ++it)
	{
		int CurPtIdx = pt*ICP_DATA_DIM;

		PtsPtr[CurPtIdx+3] = it.Value()[0]; // R
		PtsPtr[CurPtIdx+4] = it.Value()[1]; // G
		PtsPtr[CurPtIdx+5] = it.Value()[2]; // B
	}

	// Synchronize colors to GPU
	m_Pts->SynchronizeDevice();

	// Now copy the range image to the device
	ritkCudaSafeCall(cudaMemcpyToArray(m_InputImgArr, 0, 0, Data->GetRangeImage()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT*sizeof(float), cudaMemcpyHostToDevice));

	// Compute the world coordinates
	CUDARangeToWorld(
		m_Pts->GetCudaMemoryPointer(),
		m_InputImgArr
		);

	// And synchronize transformed points (including colors) to host
	m_Pts->SynchronizeHost();

	// Now we extract m_NumLandmarks landmarks (a subset of all points)
	ExtractLandmarks();

	// And synchronize to the host
	m_Landmarks->SynchronizeHost();	
}


//----------------------------------------------------------------------------
void
KinectDataManager::SetNumberOfLandmarks(unsigned long NumLandmarks)
{
	if (m_NumLandmarks != NumLandmarks)
	{
		m_NumLandmarks = NumLandmarks;

		// Resize landmark container
		m_Landmarks = DatasetContainer::New();
		DatasetContainer::SizeType DataSize;
		DataSize.SetElement(0, ICP_DATA_DIM*m_NumLandmarks);
		m_Landmarks->SetContainerSize(DataSize);
		m_Landmarks->Reserve(DataSize[0]);

		// Resize landmark indices container
		m_LandmarkIndices = IndicesContainer::New();
		IndicesContainer::SizeType IndicesSize;
		IndicesSize.SetElement(0, m_NumLandmarks);
		m_LandmarkIndices->SetContainerSize(IndicesSize);
		m_LandmarkIndices->Reserve(IndicesSize[0]);

		// Now regenerate the array of indices for the landmarks
		UpdateLandmarkIndices();
	}
}


//----------------------------------------------------------------------------
void
KinectDataManager::SetClipPercentage(float ClipPercentage)
{
	m_ClipPercentage = ClipPercentage;

	// Now regenerate the array of indices for the landmarks
	UpdateLandmarkIndices();

	// Now we extract m_NumLandmarks landmarks (a subset of all points)
	ExtractLandmarks();

	// And synchronize to the host
	m_Landmarks->SynchronizeHost();	
}


//----------------------------------------------------------------------------
void
KinectDataManager::UpdateLandmarkIndices()
{
	// These values were determined experimentally
	int validXStart = 15;
	int validXEnd = 600;
	int validYStart = 50;
	int validYEnd = 478;

	int xDiff = validXEnd-validXStart;
	int yDiff = validYEnd-validYStart;

	validXStart += xDiff*m_ClipPercentage; 
	validXEnd -= xDiff*m_ClipPercentage;
	validYStart += yDiff*m_ClipPercentage;
	validYEnd -= yDiff*m_ClipPercentage;

	// Update values
	xDiff = validXEnd-validXStart;
	yDiff = validYEnd-validYStart;

	int nrOfValidPoints = xDiff * yDiff;
	int stepSize = nrOfValidPoints / m_NumLandmarks;
	int stepX = (double)xDiff / sqrt((double)m_NumLandmarks);
	int stepY = (double)yDiff / sqrt((double)m_NumLandmarks);

	unsigned long* IndicesPtr = m_LandmarkIndices->GetBufferPointer();
	int count = 0;
	for (int i = validYStart; i < validYEnd; i += stepY)
	{
		for (int j = validXStart; j < validXEnd; j += stepX)
		{
			IndicesPtr[count++] = i * KINECT_IMAGE_WIDTH + j;
			if (count >= m_NumLandmarks)
				break;
		}
		if (count >= m_NumLandmarks)
			break;
	}

	// Synchronize those indices to the device
	m_LandmarkIndices->SynchronizeDevice();
}


//----------------------------------------------------------------------------
void
KinectDataManager::TransformPts(MatrixContainer::Pointer Mat)
{
	// Apply transformation matrix Mat to points
	CUDATransformPoints3D(
		m_Pts->GetCudaMemoryPointer(),
		Mat->GetCudaMemoryPointer(),
		KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT,
		ICP_DATA_DIM
		);

	// Synchronize data to host
	m_Pts->SynchronizeHost();
}


//----------------------------------------------------------------------------
void
KinectDataManager::SwapPointsContainer(KinectDataManager* Other)
{
	// Swap points
	DatasetContainer::Pointer PtsTmp;
	PtsTmp = this->m_Pts;
	this->m_Pts = Other->m_Pts;
	Other->m_Pts = PtsTmp;
}


//----------------------------------------------------------------------------
void
KinectDataManager::ExtractLandmarks()
{
	CUDAExtractLandmarks(
		m_Landmarks->GetCudaMemoryPointer(),
		m_Pts->GetCudaMemoryPointer(),
		m_LandmarkIndices->GetCudaMemoryPointer(),
		m_NumLandmarks
		);
}
