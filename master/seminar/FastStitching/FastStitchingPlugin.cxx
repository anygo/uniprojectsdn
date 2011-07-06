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
void CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray, int w, int h);

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
	connect(this,											SIGNAL(LiveFastStitchingFrameAvailable()),		this, SLOT(LoadStitch()));

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

	m_PreviousTransform = vtkSmartPointer<vtkMatrix4x4>::New();
	m_PreviousTransform->Identity();

	// dirty hack - we just initialized m_icp and m_cpf, but the plugin crashes if we don't do it again (Qt contexts?!)
	m_ResetICPandCPFRequired = true;
}

FastStitchingPlugin::~FastStitchingPlugin()
{
	//std::cout << "~FastStitchingPlugin" << std::endl;

	delete[] m_RangeTextureData;
	delete[] m_WCs;

	if(m_LMIndices) delete[] m_LMIndices;
	if(m_ClippedLMIndices) delete[] m_ClippedLMIndices;

	// Free GPU Memory that holds the previous and current world data
	cutilSafeCall(cudaFreeArray(m_InputImgArr));
	cutilSafeCall(cudaFree(m_devWCs));

	if(m_devSourceIndices) cutilSafeCall(cudaFree(m_devSourceIndices));
	if(m_devTargetIndices) cutilSafeCall(cudaFree(m_devTargetIndices));
	if(m_devSourceLandmarks) cutilSafeCall(cudaFree(m_devSourceLandmarks));
	if(m_devTargetLandmarks) cutilSafeCall(cudaFree(m_devTargetLandmarks));

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
				emit LiveFastStitchingFrameAvailable();
			}

			// unlock mutex
			m_Mutex.unlock();
		}

		emit UpdateGUI();
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::LoadStitch()
{
	//std::cout << "LoadStitch" << std::endl;

	QTime t;

	if (m_ResetICPandCPFRequired)
		Reset();

	// load the new frame
	t.start();
	LoadFrame();
	std::cout << "LoadFrame in " << t.elapsed() << " ms" << std::endl;

	static bool firstFrame = true;
	if (!firstFrame)
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
	VisualizeFrame();
	std::cout << "VisualizeFrame in " << t.elapsed() << " ms" << std::endl;

	// swap buffers
	float4* tmp = m_devWCs;
	m_devWCs = m_devPrevWCs;
	m_devPrevWCs = tmp;

	firstFrame = false;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::VisualizeFrame()
{
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

	// update m_Data
	polyData->SetPoints(points);
	polyData->GetPointData()->SetScalars(colors);
	polyData->SetVerts(cells);
	polyData->Update();


	// VISUALIZATION BUFFER
	const int bufSize = 10;

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
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::LoadFrame()
{
	// Copy the input data to the device
	cutilSafeCall(cudaMemcpyToArray(m_InputImgArr, 0, 0, m_CurrentFrame->GetRangeImage()->GetBufferPointer(), FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float), cudaMemcpyHostToDevice));

	// Compute the world coordinates
	CUDARangeToWorld(m_devWCs, m_InputImgArr, FRAME_SIZE_X, FRAME_SIZE_Y);
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::ExtractLandmarks()
{
	//std::cout << "ExtractLandmarks" << std::endl;

	CUDAExtractLandmarks(m_NumLandmarks, m_devWCs, m_devSourceIndices, m_devSourceLandmarks);
	CUDAExtractLandmarks(m_NumLandmarks, m_devPrevWCs, m_devTargetIndices, m_devTargetLandmarks);
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::ResetICPandCPF() 
{
	//std::cout << "ResetICPandCPF" << std::endl;

	m_ResetICPandCPFRequired = true;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::Reset() 
{
	m_NumLandmarks = m_Widget->m_SpinBoxLandmarks->value();

	if(m_ClippedLMIndices) delete[] m_ClippedLMIndices;
	if(m_LMIndices) delete[] m_LMIndices;

	m_ClippedLMIndices = new unsigned int[m_NumLandmarks];
	m_LMIndices = new unsigned int[m_NumLandmarks];

	if(m_devSourceIndices) cutilSafeCall(cudaFree(m_devSourceIndices));
	if(m_devTargetIndices) cutilSafeCall(cudaFree(m_devTargetIndices));
	if(m_devSourceLandmarks) cutilSafeCall(cudaFree(m_devSourceLandmarks));
	if(m_devTargetLandmarks) cutilSafeCall(cudaFree(m_devTargetLandmarks));

	cutilSafeCall(cudaMalloc((void**)&(m_devSourceIndices), m_NumLandmarks*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devTargetIndices), m_NumLandmarks*sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&(m_devSourceLandmarks), m_NumLandmarks*sizeof(float4)));
	cutilSafeCall(cudaMalloc((void**)&(m_devTargetLandmarks), m_NumLandmarks*sizeof(float4)));


	m_cpf = new ClosestPointFinderRBCGPU(m_NumLandmarks, m_Widget->m_DoubleSpinBoxRGBWeight->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value()));

	delete m_icp;
	m_icp = new ExtendedICPTransform;
	m_icp->SetClosestPointFinder(m_cpf);
	m_icp->SetNumLandmarks(m_NumLandmarks);

	// create source and target indices and copy to gpu
	unsigned int* srcIndices = new unsigned int[m_NumLandmarks];
	int stepSize = (FRAME_SIZE_X * FRAME_SIZE_Y) / m_NumLandmarks;

	for (int i = 0, j = 0; i < m_NumLandmarks; ++i, j += stepSize)
	{
		srcIndices[i] = j;
	}

	// atm source and target indices are the same... no clipping! TODO!!!
	cutilSafeCall(cudaMemcpy(m_devSourceIndices, srcIndices, m_NumLandmarks*sizeof(unsigned int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(m_devTargetIndices, srcIndices, m_NumLandmarks*sizeof(unsigned int), cudaMemcpyHostToDevice));

	delete[] srcIndices;

	m_PreviousTransform = vtkSmartPointer<vtkMatrix4x4>::New();
	m_PreviousTransform->Identity();

	m_ResetICPandCPFRequired = false;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::Stitch()
{
	//std::cout << "Stitch" << std::endl;
	m_icp->SetSource(m_devSourceLandmarks);
	m_icp->SetTarget(m_devTargetLandmarks);
	m_icp->SetMaxMeanDist(static_cast<float>(m_Widget->m_DoubleSpinBoxMaxRMS->value()));
	m_icp->SetMaxIter(m_Widget->m_SpinBoxMaxIterations->value());

	// transform the landmarks with previous transformation matrix
	CUDATransformPoints(m_PreviousTransform->Element, m_devSourceLandmarks, m_NumLandmarks, NULL);

	// and to all WCs TODO: combine with later transform
	CUDATransformPoints(m_PreviousTransform->Element, m_devWCs, m_NumLandmarks, NULL);

	// new stuff... strange
	//double* bounds = toBeStitched->GetBounds();
	//double boundDiagonal = sqrt((bounds[1] - bounds[0])*(bounds[1] - bounds[0]) + (bounds[3] - bounds[2])*(bounds[3] - bounds[2]) + (bounds[5] - bounds[4])*(bounds[5] - bounds[4]));
	//m_icp->SetNormalizeRGBToDistanceValuesFactor(static_cast<float>(boundDiagonal / sqrt(3.0)));

	// Start ICP and yield final transformation matrix
	m_PreviousTransform->DeepCopy(m_icp->StartICP());


	// perform the transform on GPU (m_PreviousTransform is now the current transform)
	CUDATransformPoints(m_PreviousTransform->Element, m_devWCs, FRAME_SIZE_X * FRAME_SIZE_Y, NULL);

	// update debug information in GUI
	m_Widget->m_LabelICPIterations->setText(QString::number(m_icp->GetNumIter()));
	m_Widget->m_LabelICPError->setText(QString::number(m_icp->GetMeanDist()));
}