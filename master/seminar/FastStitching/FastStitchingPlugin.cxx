// standard includes
#include "FastStitchingPlugin.h"
#include "DebugManager.h"
#include "Manager.h"

// cpp std
#include <vector>

// Qt includes
#include <QFileDialog>
#include <QTime>
#include <QColorDialog>
#include <QMessageBox>

// VTK includes
#include <vtkPointData.h>
#include <vtkPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkCleanPolyData.h>
#include <vtkDelaunay2D.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkLandmarkTransform.h>
#include <vtkAppendPolyData.h>
#include <vtkBox.h>
#include <vtkClipPolyData.h>
#include <vtkProperty.h>
#include <vtkRendererCollection.h>
#include <vtkUnsignedCharArray.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

// our includes
#include <ClosestPointFinderRBCGPU.h>
#include <defs.h>

extern "C"
void CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray);

extern "C"
void CUDAExtractLandmarks(int numLandmarks, float4* devWCsIn, unsigned int* devIndicesIn, float4* devLandmarksOut);

extern "C"
void CUDATransformPoints(double transformationMatrix[4][4], float4* toBeTransformed, int numPoints, float* distances);


FastStitchingPlugin::FastStitchingPlugin()
{
	// create the widget
	m_Widget = new FastStitchingWidget();

	// basic signals
	connect(this, SIGNAL(UpdateGUI()), m_Widget,							SLOT(UpdateGUI()));
	connect(this, SIGNAL(UpdateGUI()), m_Widget->m_VisualizationWidget3D,	SLOT(UpdateGUI()));

	// our signals and slots
	connect(m_Widget->m_PushButtonStitchFrame,				SIGNAL(clicked()),								this, SLOT(LoadStitch()));
	connect(m_Widget->m_SpinBoxLandmarks,					SIGNAL(valueChanged(int)),						this, SLOT(ResetICPandCPF()));
	connect(m_Widget->m_DoubleSpinBoxRGBWeight,				SIGNAL(valueChanged(double)),					this, SLOT(ResetICPandCPF()));
	connect(this,											SIGNAL(NewFrameAvailable()),					this, SLOT(LoadStitch()));

	m_NumLandmarks = m_Widget->m_SpinBoxLandmarks->value();

	// initialize member objects
	m_RangeTextureData = new unsigned char[FRAME_SIZE_X*FRAME_SIZE_Y];
	m_WCs = new float4[FRAME_SIZE_X*FRAME_SIZE_Y];

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cutilSafeCall(cudaMallocArray(&m_InputImgArr, &ChannelDesc, FRAME_SIZE_X, FRAME_SIZE_Y));

	cutilSafeCall(cudaMalloc((void**)&(m_devWCs), FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float4)));
	cutilSafeCall(cudaMalloc((void**)&(m_devPrevWCs), FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float4)));
	
	m_FramesProcessed = 0;

	// iterative closest point (ICP) transformation
	m_icp = new ExtendedICPTransform;

	// initialize ClosestPointFinder
	m_cpf = new ClosestPointFinderRBCGPU(m_NumLandmarks, m_Widget->m_DoubleSpinBoxRGBWeight->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value()));

	// comparison of two consecutive frames
	m_LMIndices = NULL;
	m_ClippedLMIndices = NULL;
	m_devSourceIndices = NULL;
	m_devTargetIndices = NULL;
	m_devSourceLandmarks = NULL;
	m_devTargetLandmarks = NULL;
	m_SrcIndices = NULL;
	m_TargetIndices = NULL;
	m_LandmarksColor = NULL;

	m_PreviousTransform = vtkSmartPointer<vtkMatrix4x4>::New();
	m_PreviousTransform->Identity();

	// dirty hack - we just initialized m_icp and m_cpf, but the plugin crashes if we don't do it again (Qt contexts?!)
	m_ResetICPandCPFRequired = true;
	m_FirstFrame = true;
}

FastStitchingPlugin::~FastStitchingPlugin()
{
	//std::cout << "~FastStitchingPlugin" << std::endl;

	delete[] m_RangeTextureData;
	delete[] m_WCs;

	if(m_LMIndices) delete[] m_LMIndices;
	if(m_ClippedLMIndices) delete[] m_ClippedLMIndices;
	if(m_SrcIndices) delete[] m_SrcIndices;
	if(m_TargetIndices) delete[] m_TargetIndices;
	if(m_LandmarksColor) delete[] m_LandmarksColor;

	// Free GPU Memory that holds the previous and current world data
	cutilSafeCall(cudaFreeArray(m_InputImgArr));
	cutilSafeCall(cudaFree(m_devWCs));

	if(m_devSourceIndices) cutilSafeCall(cudaFree(m_devSourceIndices));
	if(m_devTargetIndices) cutilSafeCall(cudaFree(m_devTargetIndices));
	if(m_devSourceLandmarks) cutilSafeCall(cudaFree(m_devSourceLandmarks));
	if(m_devTargetLandmarks) cutilSafeCall(cudaFree(m_devTargetLandmarks));
	if(m_devCurLandmarksColor) cutilSafeCall(cudaFree(m_devCurLandmarksColor));
	if(m_devPrevLandmarksColor) cutilSafeCall(cudaFree(m_devPrevLandmarksColor));

	// delete ClosestPointFinder
	delete m_cpf;
	delete m_Widget;
}
//----------------------------------------------------------------------------
QString
FastStitchingPlugin::GetName()
{
	return tr("FastStitchingPlugin");
}
//----------------------------------------------------------------------------
QWidget*
FastStitchingPlugin::GetPluginGUI()
{
	return m_Widget;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::ProcessEvent(ritk::Event::Pointer EventP)
{
	// New frame event
	if (EventP->type() == ritk::NewFrameEvent::EventType)
	{
		++m_FramesProcessed;

		// skip frame if plugin is still working
		if (m_Mutex.tryLock())
		{
			// Cast the event
			ritk::NewFrameEvent::Pointer NewFrameEventP = qSharedPointerDynamicCast<ritk::NewFrameEvent, ritk::Event>(EventP);
			if ( !NewFrameEventP )
			{
				LOG_DEB("Event mismatch detected: Type=" << EventP->type());
				return;
			}

			m_CurrentFrame = NewFrameEventP->RImage;


			// run autostitching for each frame if checkbox is checked
			if (m_Widget->m_RadioButtonLiveFastStitching->isChecked() && m_FramesProcessed % m_Widget->m_SpinBoxFrameStep->value() == 0)
			{
				emit NewFrameAvailable();
			}

			// unlock mutex
			m_Mutex.unlock();
		}
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::Reset() 
{
	m_NumLandmarks = m_Widget->m_SpinBoxLandmarks->value();

	if(m_ClippedLMIndices) delete[] m_ClippedLMIndices;
	if(m_LMIndices) delete[] m_LMIndices;
	if(m_SrcIndices) delete[] m_SrcIndices;
	if(m_TargetIndices) delete[] m_TargetIndices;
	if(m_LandmarksColor) delete[] m_LandmarksColor;

	
	m_SrcIndices = new unsigned int[m_NumLandmarks];
	m_TargetIndices = new unsigned int[m_NumLandmarks];

	m_ClippedLMIndices = new unsigned int[m_NumLandmarks];
	m_LMIndices = new unsigned int[m_NumLandmarks];
	m_LandmarksColor = new float4[m_NumLandmarks];

	if(m_devSourceIndices) cutilSafeCall(cudaFree(m_devSourceIndices));
	if(m_devTargetIndices) cutilSafeCall(cudaFree(m_devTargetIndices));
	if(m_devSourceLandmarks) cutilSafeCall(cudaFree(m_devSourceLandmarks));
	if(m_devTargetLandmarks) cutilSafeCall(cudaFree(m_devTargetLandmarks));
	if(m_devCurLandmarksColor) cutilSafeCall(cudaFree(m_devCurLandmarksColor));
	if(m_devPrevLandmarksColor) cutilSafeCall(cudaFree(m_devPrevLandmarksColor));

	cutilSafeCall(cudaMalloc((void**)&(m_devSourceIndices), m_NumLandmarks*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devTargetIndices), m_NumLandmarks*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devSourceLandmarks), m_NumLandmarks*sizeof(float4)));
	cutilSafeCall(cudaMalloc((void**)&(m_devTargetLandmarks), m_NumLandmarks*sizeof(float4)));
	// TODO COLORS
	cutilSafeCall(cudaMalloc((void**)&(m_devCurLandmarksColor), m_NumLandmarks*sizeof(float4)));
	cutilSafeCall(cudaMalloc((void**)&(m_devPrevLandmarksColor), m_NumLandmarks*sizeof(float4)));


	m_cpf = new ClosestPointFinderRBCGPU(m_NumLandmarks, m_Widget->m_DoubleSpinBoxRGBWeight->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value()));

	delete m_icp;
	m_icp = new ExtendedICPTransform;
	m_icp->SetClosestPointFinder(m_cpf);
	m_icp->SetNumLandmarks(m_NumLandmarks);


	int validXStart = 15;
	int validXEnd = 600;
	int validYStart = 50;
	int validYEnd = 478;

	int nrOfValidPoints = (validXEnd - validXStart) * (validYEnd - validYStart);
	int stepSize = nrOfValidPoints / m_NumLandmarks;

	for (int i = 0, j = validYStart*FRAME_SIZE_X + validXStart; i < m_NumLandmarks; ++i, j += stepSize)
	{
		if(j % FRAME_SIZE_X > validXEnd)
			j += FRAME_SIZE_X - validXEnd + validXStart;

		targetIndices[i] = j;
	}

	double clipPercentage = m_Widget->m_DoubleSpinBoxClipPercentage->value();

	validXStart += (validXEnd-validXStart)*clipPercentage; 
	validXEnd -= (validXEnd-validXStart)*clipPercentage;
	validYStart += (validYEnd-validYStart)*clipPercentage;
	validYEnd -= (validYEnd-validYStart)*clipPercentage;

	nrOfValidPoints = (validXEnd - validXStart) * (validYEnd - validYStart);
	stepSize = nrOfValidPoints / m_NumLandmarks;

	for (int i = 0, j = validYStart*FRAME_SIZE_X + validXStart; i < m_NumLandmarks; ++i, j += stepSize)
	{
		if(j % FRAME_SIZE_X > validXEnd)
			j += FRAME_SIZE_X - validXEnd + validXStart;

		srcIndices[i] = j;
	}

	cutilSafeCall(cudaMemcpy(m_devSourceIndices, srcIndices, m_NumLandmarks*sizeof(unsigned int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(m_devTargetIndices, targetIndices, m_NumLandmarks*sizeof(unsigned int), cudaMemcpyHostToDevice));

	m_PreviousTransform = vtkSmartPointer<vtkMatrix4x4>::New();
	m_PreviousTransform->Identity();

	m_ResetICPandCPFRequired = false;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::LoadStitch()
{
	size_t freeMemory;
	size_t totalMemory;
	cudaMemGetInfo(&freeMemory, &totalMemory);
	printf("Total memory: %lu\tFree memory: %lu\n",
           (unsigned long) totalMemory / 1024 / 1024,
           (unsigned long) freeMemory / 1024 / 1024);

	QTime t;

	if (m_ResetICPandCPFRequired)
		Reset();

	// load the new frame
	t.start();
	LoadFrame();
	std::cout << "LoadFrame in " << t.elapsed() << " ms" << std::endl;

	if (!m_FirstFrame)
	{
		// now we have to extract the landmarks
		t.start();
		ExtractLandmarks();
		std::cout << "ExtractLandmarks in " << t.elapsed() << " ms" << std::endl;

		// stitch to just loaded frame to the previous frame (given by last history entry)
		t.start();
		Stitch();
		std::cout << "Stitch in " << t.elapsed() << " ms" << std::endl;
	}

	// Visualize frame
	t.start();
	CopyToCPUAndVisualizeFrame();
	std::cout << "CopyToCPUAndVisualizeFrame in " << t.elapsed() << " ms" << std::endl;

	// swap buffers
	float4* tmp = m_devWCs;
	m_devWCs = m_devPrevWCs;
	m_devPrevWCs = tmp;

	// TODO swap the landmark colors
	tmp = m_devCurLandmarksColor;
	m_devCurLandmarksColor = m_devPrevLandmarksColor;
	m_devPrevLandmarksColor = tmp;


	m_FirstFrame = false;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::LoadFrame()
{
	// Copy the input data to the device
	cutilSafeCall(cudaMemcpyToArray(m_InputImgArr, 0, 0, m_CurrentFrame->GetRangeImage()->GetBufferPointer(), FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float), cudaMemcpyHostToDevice));

	// TODO: Also copy  m_CurrentFrame->GetRGBImage()->GetBufferPointer() to the GPU, either texture or whatever memory
	

	// Compute the world coordinates
	CUDARangeToWorld(m_devWCs, m_InputImgArr);

	// Extract color
	//typedef itk::ImageRegionConstIterator<RImageType::RGBImageType> IteratorType;
	//IteratorType it(m_CurrentFrame->GetRGBImage(), m_CurrentFrame->GetRGBImage()->GetRequestedRegion());

	//it.GoToBegin();
	//for (int i = 0; i < m_NumLandmarks; ++i)
	//{
	//	it.SetIndex(m_SrcIndices[i]);
	//	m_LandmarksColor[i] = make_float4(it.Value()[0], it.Value()[1], it.Value()[2], 1.);
	//}

	//cutilSafeCall(cudaMemcpy(m_devCurLandmarksColor, m_LandmarksColor, m_NumLandmarks*sizeof(float4), cudaMemcpyHostToDevice));

}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::ExtractLandmarks()
{
	// TODO: Change Function that it parallely also extracs color
	CUDAExtractLandmarks(m_NumLandmarks, m_devWCs, m_devSourceIndices, m_devSourceLandmarks);
	CUDAExtractLandmarks(m_NumLandmarks, m_devPrevWCs, m_devTargetIndices, m_devTargetLandmarks);
	
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::Stitch()
{
	m_icp->SetSource(m_devSourceLandmarks, m_devCurLandmarksColor);
	m_icp->SetTarget(m_devTargetLandmarks, m_devCurLandmarksColor);

	m_icp->SetMaxMeanDist(static_cast<float>(m_Widget->m_DoubleSpinBoxMaxRMS->value()));
	m_icp->SetMaxIter(m_Widget->m_SpinBoxMaxIterations->value());

	// transform the landmarks with previous transformation matrix
	CUDATransformPoints(m_PreviousTransform->Element, m_devSourceLandmarks, m_NumLandmarks, NULL);

	// new stuff... strange
	//double* bounds = toBeStitched->GetBounds();
	//double boundDiagonal = sqrt((bounds[1] - bounds[0])*(bounds[1] - bounds[0]) + (bounds[3] - bounds[2])*(bounds[3] - bounds[2]) + (bounds[5] - bounds[4])*(bounds[5] - bounds[4]));
	//m_icp->SetNormalizeRGBToDistanceValuesFactor(static_cast<float>(boundDiagonal / sqrt(3.0)));

	// Start ICP and yield final transformation matrix
	vtkMatrix4x4* icpMatrix = m_icp->StartICP();

	// print matrix
	//icpMatrix->Print(std::cout);
	//m_PreviousTransform->Print(std::cout);

	// combine previous transform with estimated transform to transform all WCs
	vtkMatrix4x4::Multiply4x4(icpMatrix, m_PreviousTransform, m_PreviousTransform);

	// perform the transform on GPU (m_PreviousTransform is now the current transform)
	CUDATransformPoints(m_PreviousTransform->Element, m_devWCs, FRAME_SIZE_X * FRAME_SIZE_Y, NULL);

	// update debug information in GUI
	m_Widget->m_LabelICPIterations->setText(QString::number(m_icp->GetNumIter()));
	m_Widget->m_LabelICPError->setText(QString::number(m_icp->GetMeanDist()));
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::CopyToCPUAndVisualizeFrame()
{
	// copy transformed WC data from GPU to CPU
	cutilSafeCall(cudaMemcpy(m_WCs, m_devWCs, FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float4), cudaMemcpyDeviceToHost));


	if (!m_Widget->m_CheckBoxShowFrames->isChecked())
		return;

	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();

	vtkSmartPointer<vtkDataArray> colors =
		vtkSmartPointer<vtkUnsignedCharArray>::New();
	colors->SetNumberOfComponents(4);

	vtkSmartPointer<vtkCellArray> cells =
		vtkSmartPointer<vtkCellArray>::New();

	typedef itk::ImageRegionConstIterator<RImageType::RGBImageType> IteratorType;
	IteratorType it(m_CurrentFrame->GetRGBImage(), m_CurrentFrame->GetRGBImage()->GetRequestedRegion());

	float4 p;
	it.GoToBegin();

	for (int i = 0; i < FRAME_SIZE_X*FRAME_SIZE_Y; ++i, ++it)
	{
		p = m_WCs[i];

		if (p.x == p.x) // i.e. not QNAN
		{
			points->InsertNextPoint(p.x, p.y, p.z);
			float r = it.Value()[0];
			float g = it.Value()[1];
			float b = it.Value()[2];
			colors->InsertNextTuple4(r, g, b, 255);
		}
	}

	for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
	{
		cells->InsertNextCell(1, &i);
	}	

	vtkSmartPointer<vtkPolyData> polyData =
		vtkSmartPointer<vtkPolyData>::New();

	// create polydata
	polyData->SetPoints(points);
	polyData->GetPointData()->SetScalars(colors);
	polyData->SetVerts(cells);
	polyData->Update();


	// VISUALIZATION BUFFER
	const int bufSize = 50;

	static int bufCtr = 0;
	static vtkSmartPointer<ritk::RImageActorPipeline> actors[bufSize];
	
	if (bufCtr < bufSize)
	{
		actors[bufCtr] = vtkSmartPointer<ritk::RImageActorPipeline>::New();
		actors[bufCtr]->SetData(polyData, true);
		m_Widget->m_VisualizationWidget3D->AddActor(actors[bufCtr]);
	} else
	{
		actors[bufCtr % bufSize]->SetData(polyData, true);
	}

	++bufCtr;

	emit UpdateGUI();
}