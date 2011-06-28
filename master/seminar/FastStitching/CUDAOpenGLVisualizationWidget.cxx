#include "CUDAOpenGLVisualizationWidget.h"

#include "OpenGLExtensions.h"
#include <cutil_inline.h>
#include <cuda_gl_interop.h>

#include <gl/gl.h>
#include <gl/glu.h>
#include "Shaders.h"
#include "LUTs.h"

#include <math.h>
#include <algorithm>

#include <QTime>
#include <vtkPoints.h>
#include <vtkLandmarkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>


#include "Manager.h"
#include "CudaRegularMemoryImportImageContainer.h"


extern "C"
void
CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray, int w, int h);

extern "C"
void
CUDAFindLandmarks(float4* source, float4* target, float4* source_out, float4* target_out, int* indices_source, int* indices_target, int numLandmarks);

extern "C"
void
CUDAFindNNBF(float4* source, float4* target, int* correspondences, int numLandmarks);

extern "C"
void
CUDATransfromLandmarks(float4* toBeTransformed, double matrix[4][4], int numLandmarks);


CUDAOpenGLVisualizationWidget::CUDAOpenGLVisualizationWidget(QWidget *parent) :
QGLWidget(parent)
{
	//cutilSafeCall(cudaThreadExit());
	cutilSafeCall(cudaSetDevice(cutGetMaxGflopsDeviceId()));
	cutilSafeCall(cudaGLSetGLDevice(cutGetMaxGflopsDeviceId()));
	m_AllocatedSize = 0;
	m_InputImgArr = NULL;
	m_Output = NULL;

	m_CurWCs = NULL;
	m_PrevWCs = NULL;

	m_renderPoints = true;

	// To accept key events
	setFocusPolicy(Qt::StrongFocus);

	// Register this smart pointer
	qRegisterMetaType<ritk::RImageF2::ConstPointer>("ritk::RImageF2::ConstPointer");

	// No data available
	m_CurrentFrame = NULL;

	// No texture so far
	m_TextureSize[0] = 0;
	m_TextureSize[1] = 0;
	m_TextureCoords = 0;
	m_RGBTextureData = 0;
	m_RangeTextureData = 0;

	m_ClippingPlanes[0] = 0;
	m_ClippingPlanes[1] = 0;

	// No translation by default
	m_Translation[0] = 0;
	m_Translation[1] = 0;

	// No rotation by default
	m_Rotation[0] = 0;
	m_Rotation[1] = 0;
	m_Rotation[2] = 0;

	// Std zoom
	m_RawZoom = 0.5f*log(0.5f/1.5f); // = atanh(y/2+1) 
	m_Zoom = (tanh(m_RawZoom)+1)*2;

	m_Counters[0] = 0;
	m_Counters[1] = 0;
	m_Counters[2] = 0;
	m_Counters[3] = 0;
	m_Timers[0] = 0;
	m_Timers[1] = 1;
	m_Timers[2] = 1;
	m_Timers[3] = 1;

	// Shader
	m_LUTID = 0;
	m_Alpha = 0.0;
	m_ShaderProgram = 0;

	// Internal communication
	connect(this, SIGNAL(NewDataAvailable(bool)), this, SLOT(UpdateVBO(bool)), Qt::BlockingQueuedConnection);


	m_PrevTrans = vtkSmartPointer<vtkMatrix4x4>::New();
	m_PrevTrans->Identity();
}


//----------------------------------------------------------------------------
CUDAOpenGLVisualizationWidget::~CUDAOpenGLVisualizationWidget()
{
	// Clean up
	ritk::glDeleteBuffers(1, &m_VBOVertices);

	// Free GPU Memory that holds the previous and current world data
	cutilSafeCall(cudaFree(m_CurWCs));
	cutilSafeCall(cudaFree(m_PrevWCs));

	if ( m_TextureCoords )
		delete[] m_TextureCoords;

	if ( m_RGBTextureData )
		delete[] m_RGBTextureData;

	if ( m_RangeTextureData )
		delete[] m_RangeTextureData;

	if ( m_InputImgArr )
		cutilSafeCall(cudaFreeArray(m_InputImgArr));
}

//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::Prepare(ritk::RImageF2::ConstPointer Data)
{
	// Lock ourself
	m_Mutex.lock();

	// Initialize timer
	LONGLONG Frequency;
	QueryPerformanceFrequency((LARGE_INTEGER*)&Frequency);

	LONGLONG C1;
	QueryPerformanceCounter((LARGE_INTEGER*)&C1);

	bool ResetCameraRequired = false;
	if ( !m_CurrentFrame )
	{
		ResetCameraRequired = true;
	}

	// Update the current frame
	m_CurrentFrame = Data;

	// For convenience
	long SizeX = m_CurrentFrame->GetBufferedRegion().GetSize()[0];
	long SizeY = m_CurrentFrame->GetBufferedRegion().GetSize()[1];

	// Flag that indicates whether the texture size changed
	bool SizeChanged = SizeX != m_TextureSize[0] || SizeY != m_TextureSize[1];

	// Prepare the texture coords
	if (SizeChanged)
	{
		// Delete old memory and allocate memory to fit the requirements
		if ( m_TextureCoords )
			delete[] m_TextureCoords;

		m_TextureCoords = new GLfloat[SizeX*SizeY*2 + SizeX*(SizeY-2)*2];

		// Update coords
		for ( long l = 0; l < SizeX*SizeY + SizeX*(SizeY-2); l++ )
		{
			if (l%2==0)
			{
				m_TextureCoords[l*2+0] = (l%(SizeX*2))/((SizeX*2)*1.f);
				m_TextureCoords[l*2+1] = (l/(SizeX*2))/(SizeY*1.f) ;
			}
			else
			{
				m_TextureCoords[l*2+0] = (l%(SizeX*2)-1)/((SizeX*2)*1.f);
				m_TextureCoords[l*2+1] = (l/(SizeX*2)+1)/(SizeY*1.f);
			}
		}

	}

	// Prepare the rgb texture data
	if (SizeChanged)
	{
		// Delete old memory and allocate memory to fit the requirements
		if ( m_RGBTextureData )
			delete[] m_RGBTextureData;
		m_RGBTextureData = new unsigned char[SizeX*SizeY*3];
	}

	// Copy the RGB texture
	const ritk::RImageF2::RGBType *RGBData = m_CurrentFrame->GetRGBImage()->GetBufferPointer();
	unsigned char *RGBTexturePtr = m_RGBTextureData;
	for ( int l = 0; l < SizeX*SizeY; ++l, ++RGBTexturePtr, ++RGBData )
	{
		*RGBTexturePtr = (unsigned char)(*RGBData)[0];
		*(++RGBTexturePtr) = (unsigned char)(*RGBData)[1];
		*(++RGBTexturePtr) = (unsigned char)(*RGBData)[2];
	}	

	// Prepare the range texture data
	if (SizeChanged)
	{
		// Delete old memory and allocate memory to fit the requirements
		if ( m_RangeTextureData )
			delete[] m_RangeTextureData;
		m_RangeTextureData = new unsigned char[SizeX*SizeY];
	}

	// Clamp and rescale the range information
	unsigned char *RTexturePtr = m_RangeTextureData;
	const float *RData = m_CurrentFrame->GetRangeImage()->GetBufferPointer();
	float Scale = 1.f/(m_RangeBoundaries[1]-m_RangeBoundaries[0]);

	for ( long l = 0; l < SizeX*SizeY; l++, RTexturePtr++ )
	{
		float val = RData[l];
		if ( val < m_RangeBoundaries[0] )
			val = m_RangeBoundaries[0];
		if ( val > m_RangeBoundaries[1] )
			val = m_RangeBoundaries[1];
		*RTexturePtr = (unsigned char)(((val - m_RangeBoundaries[0])*Scale)*255);
	}

	// Remember the texture size
	m_TextureSize[0] = SizeX;
	m_TextureSize[1] = SizeY;


	// Unlock ourself
	m_Mutex.unlock();

	// Delegate the data to the UpdateVBO slot that is connected to the NewDataAvailable signal.
	emit NewDataAvailable(SizeChanged);
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::Stitch() 
{
	m_Mutex.lock();

	QTime tOverall;
	tOverall.start();

	QTime t;
	t.start();

	// compute sampling grid;
	const int numLandmarks = 3000;
	const int clipSize = 50;

	int indices_source[numLandmarks];
	int indices_target[numLandmarks];

	// without clipping - just use each step'th index
	int numPoints = m_TextureSize[0] * m_TextureSize[1];
	int step = numPoints / numLandmarks;
	for (int i = 0, cur = 0; i < numLandmarks; ++i, cur += step)
	{
		indices_target[i] = cur;
	}
	
	// with clipping - some additional stuff required
	int numPointsClipped = (m_TextureSize[0] - 2*clipSize) * (m_TextureSize[1] - 2*clipSize);
	int stepClipped = numPointsClipped / numLandmarks;
	for (int i = 0, cur = m_TextureSize[0] * clipSize + clipSize; i < numLandmarks; ++i, cur += stepClipped)
	{
		if (cur % (m_TextureSize[0] - clipSize) < clipSize)
			cur += 2*clipSize;
		indices_source[i] = cur;
	}

	int* dev_indices_source;
	cutilSafeCall(cudaMalloc((void**)&(dev_indices_source), numLandmarks*sizeof(int)));
	cutilSafeCall(cudaMemcpy(dev_indices_source, indices_source, numLandmarks*sizeof(int), cudaMemcpyHostToDevice));

	int* dev_indices_target;
	cutilSafeCall(cudaMalloc((void**)&(dev_indices_target), numLandmarks*sizeof(int)));
	cutilSafeCall(cudaMemcpy(dev_indices_target, indices_target, numLandmarks*sizeof(int), cudaMemcpyHostToDevice));

	float4 source[numLandmarks];
	float4 target[numLandmarks];
	float4* dev_source;
	float4* dev_target;
	cutilSafeCall(cudaMalloc((void**)&(dev_source), numLandmarks*sizeof(float4)));
	cutilSafeCall(cudaMalloc((void**)&(dev_target), numLandmarks*sizeof(float4)));

	std::cout << "initialized in " << t.elapsed() << " ms" << std::endl;
	t.start();
	
	// Initialized Source/Target Landmarks
	CUDAFindLandmarks(m_CurWCs, m_PrevWCs, dev_source, dev_target, dev_indices_source, dev_indices_target, numLandmarks);
	
	std::cout << "CUDAFindLandmarks in " << t.elapsed() << " ms" << std::endl;
	t.start();

	// create landmarkTransform only once
	vtkLandmarkTransform* landmarkTransform = vtkLandmarkTransform::New();
	landmarkTransform->SetModeToRigidBody();
	vtkTransform* accumulate = vtkTransform::New();
	accumulate->PostMultiply();

	// copy target points only once
	cutilSafeCall(cudaMemcpy(target, dev_target, numLandmarks*sizeof(float4), cudaMemcpyDeviceToHost));

	// apply previous transform (init icp)
	CUDATransfromLandmarks(dev_source, m_PrevTrans->Element, numLandmarks);
	accumulate->Concatenate(m_PrevTrans);

	std::cout << "previousTransform in " << t.elapsed() << " ms" << std::endl;

	int correspondences[numLandmarks];
	int* dev_correspondences;
	cutilSafeCall(cudaMalloc((void**)&(dev_correspondences), numLandmarks*sizeof(int)));

	
	t.start();

	const int numIter = 50;

	// Start iterating...
	for(int k = 0; k < numIter; ++k)
	{
		// find nearest neighbors
		CUDAFindNNBF(dev_source, dev_target, dev_correspondences, numLandmarks);

		cutilSafeCall(cudaMemcpy(source, dev_source, numLandmarks*sizeof(float4), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(correspondences, dev_correspondences, numLandmarks*sizeof(int), cudaMemcpyDeviceToHost));
		
		vtkPoints* p1 = vtkPoints::New();
		vtkPoints* p2 = vtkPoints::New();

		for (int i = 0; i < numLandmarks; ++i)
		{
			if (correspondences[i] == -1)
				continue;

			p1->InsertNextPoint(source[i].x, source[i].y, source[i].z);
			p2->InsertNextPoint(target[correspondences[i]].x, target[correspondences[i]].y, target[correspondences[i]].z);	
		}

		landmarkTransform->SetSourceLandmarks(p1);
		landmarkTransform->SetTargetLandmarks(p2);
		landmarkTransform->Update();

		accumulate->Concatenate(landmarkTransform->GetMatrix());

		p2->Delete();
		p1->Delete();

		// transform points on gpu
		CUDATransfromLandmarks(dev_source, landmarkTransform->GetMatrix()->Element, numLandmarks);
	}

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << std::endl;

	std::cout << "ICP in " << t.elapsed() << " ms (avg: " << static_cast<float>(t.elapsed()) / static_cast<float>(numIter) << " ms)" << std::endl;

	t.start();
	
	// transform all points
	CUDATransfromLandmarks(m_CurWCs, accumulate->GetMatrix()->Element, numPoints);

	// update previous transform for next iteration
	vtkMatrix4x4::Multiply4x4(m_PrevTrans, accumulate->GetMatrix(), m_PrevTrans);

	float4 stitched[640*480];
	cutilSafeCall(cudaMemcpy(stitched, m_CurWCs, numPoints*sizeof(float4), cudaMemcpyDeviceToHost));

	landmarkTransform->Delete();

	cutilSafeCall(cudaFree(dev_correspondences));
	cutilSafeCall(cudaFree(dev_target));
	cutilSafeCall(cudaFree(dev_source));
	cutilSafeCall(cudaFree(dev_indices_target));
	cutilSafeCall(cudaFree(dev_indices_source));


	std::cout << "finalized in " << t.elapsed() << " ms" << std::endl;
	std::cout << "OVERALL time: " << tOverall.elapsed() << " ms" << std::endl << std::endl;

	//CreatePolyData(stitched);

	m_Mutex.unlock();

	emit FrameStitched(stitched);
}

//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::UpdateVBO(bool SizeChanged)
{
	// Lock the mutex
	m_Mutex.lock();

	// To be on the safe side
	this->makeCurrent();

	// The number of vertices that we have to process
	/*long SizeX = m_CurrentFrame->GetBufferedRegion().GetSize()[0];
	long SizeY = m_CurrentFrame->GetBufferedRegion().GetSize()[1];*/
	long SizeX = m_TextureSize[0];
	long SizeY = m_TextureSize[1];

	if(SizeX * SizeY != m_AllocatedSize)
	{
		if (m_InputImgArr)
			cutilSafeCall(cudaFreeArray(m_InputImgArr));

		cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
		cutilSafeCall(cudaMallocArray(&m_InputImgArr,&ChannelDesc,SizeX,SizeY));

		// Initialize GPU Memory to hold the previous and current world coords
		if (m_CurWCs)
			cutilSafeCall(cudaFree(m_CurWCs));
		if (m_PrevWCs)
			cutilSafeCall(cudaFree(m_PrevWCs));

		cutilSafeCall(cudaMalloc((void**)&(m_CurWCs), SizeX*SizeY*sizeof(float4)));
		cutilSafeCall(cudaMalloc((void**)&(m_PrevWCs), SizeX*SizeY*sizeof(float4)));

		m_AllocatedSize = SizeX * SizeY;
	}

	// Copy the input data to the device
	cutilSafeCall(cudaMemcpyToArray(m_InputImgArr, 0, 0, m_CurrentFrame->GetRangeImage()->GetBufferPointer(), SizeX*SizeY * sizeof(float), cudaMemcpyHostToDevice));

	// swap previous and current frame pointer s.t. the old coords don't get lost
	float4* tmp = m_CurWCs;
	m_CurWCs = m_PrevWCs;
	m_PrevWCs = tmp;

	// Compute the world coordinates
	CUDARangeToWorld
		(
		m_CurWCs,
		m_InputImgArr, 
		m_CurrentFrame->GetBufferedRegion().GetSize()[0],
		m_CurrentFrame->GetBufferedRegion().GetSize()[1]
		);

	// Unlock
	m_Mutex.unlock();
}