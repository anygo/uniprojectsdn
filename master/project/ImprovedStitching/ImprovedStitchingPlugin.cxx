#include "ImprovedStitchingPlugin.h"
#include "defs.h"

#include "ritkDebugManager.h"
#include "ritkManager.h"
#include "ritkRGBRImage.h"

#include "itkImageFileWriter.h"

#include <QFileDialog>


//----------------------------------------------------------------------------
ImprovedStitchingPlugin::ImprovedStitchingPlugin()
{
	// Create the widget
	m_Widget = new ImprovedStitchingWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget, SLOT(UpdateGUI()));

	// Plugin-specific signals/slots
	connect(m_Widget->m_ComboBoxICPNumPts, SIGNAL(currentIndexChanged(int)), this, SLOT(ChangeNumPts()));
	connect(m_Widget->m_HorizontalSliderRGBWeight, SIGNAL(valueChanged(int)), this, SLOT(ChangeRGBWeight(int)));
	connect(m_Widget->m_HorizontalSliderClipPercentage, SIGNAL(valueChanged(int)), this, SLOT(ChangeClipPercentage(int)));
	connect(m_Widget->m_PushButtonSaveVolume, SIGNAL(clicked()), this, SLOT(SaveVolume()));
	connect(m_Widget->m_PushButtonResetVolume, SIGNAL(clicked()), this, SLOT(ResetVolume()));
	connect(this, SIGNAL(FrameToStitchAvailable(bool)), this, SLOT(AutoStitch(bool)));

	// Init ICP objects to NULL
	m_ICP512   = NULL;
	m_ICP1024  = NULL;
	m_ICP2048  = NULL;
	m_ICP4096  = NULL;
	m_ICP8192  = NULL;
	m_ICP16384 = NULL;

	// Compute new weight from slider value
	m_PayloadWeight = static_cast<float>(m_Widget->m_HorizontalSliderRGBWeight->value()) / 
		static_cast<float>(m_Widget->m_HorizontalSliderRGBWeight->maximum()); // Inbetween [0;1]

	// This will set m_NumPts and init the appropriate ICP object
	ChangeNumPts();

	// Init RGB weight
	ChangeRGBWeight(m_Widget->m_HorizontalSliderRGBWeight->value());

	// Init KinectDataManagers
	m_KinectFixed = KinectDataManager::New();
	m_KinectMoving = KinectDataManager::New();

	// Init clip percentage for Kinect data manager (moving points)
	float ClipPercentage = static_cast<float>(m_Widget->m_HorizontalSliderClipPercentage->value()) / 
		static_cast<float>(m_Widget->m_HorizontalSliderClipPercentage->maximum()) / 2.01f; // Roughly inbetween [0;0.5[
	m_KinectMoving->SetClipPercentage(ClipPercentage);
	m_KinectFixed->SetClipPercentage(0);

	// -- Init VolumeManager --
	m_Volume = VolumeManager::New();

	// Set Origin
	VolumeManager::PointType Origin;
	Origin[0] = -3072;
	Origin[1] = -3072;
	Origin[2] = -3072;
	m_Volume->SetOrigin(Origin);

	// Set spacing
	VolumeManager::SpacingType Spacing;
	Spacing[0] = 24;
	Spacing[1] = 24;
	Spacing[2] = 24;
	m_Volume->SetSpacing(Spacing);

	// Set regions
	VolumeManager::RegionType Region;

	// Region requires start index and size
	VolumeManager::IndexType Start;
	Start.Fill(0);
	VolumeManager::SizeType Size;
	Size[0] = 256;
	Size[1] = 256;
	Size[2] = 256;

	Region.SetIndex(Start);
	Region.SetSize(Size);

	m_Volume->SetRegions(Region);

	// Allocate memory
	m_Volume->Allocate();

	// Init volume entity and add to visualization unit
	m_VolumeRayCastEntity = ritk::OpenGLRGBAVolumeCudaRayCastEntity::New();
	m_Widget->m_VisualizationUnit->AddEntity(m_VolumeRayCastEntity);

	// Init volume for raycasting
	m_VolumeForRaycasting = RayCastEntityType::VolumeType::New();
	m_VolumeForRaycasting->SetOrigin(m_Volume->GetOrigin());
	m_VolumeForRaycasting->SetSpacing(m_Volume->GetSpacing());
	m_VolumeForRaycasting->SetDirection(m_Volume->GetDirection());
	m_VolumeForRaycasting->SetRegions(m_Volume->GetBufferedRegion());
	m_VolumeForRaycasting->Allocate();

	Cuda3DArrayContainerType::Pointer Cuda3DArrayContainer = Cuda3DArrayContainerType::New();
	Cuda3DArrayContainer->SetContainerSize(m_VolumeForRaycasting->GetRequestedRegion().GetSize());
	m_VolumeForRaycasting->SetPixelContainer(Cuda3DArrayContainer);
	m_VolumeRayCastEntity->SetVolume(m_VolumeForRaycasting);

	m_IsFirstFrame = true;
}


//----------------------------------------------------------------------------
ImprovedStitchingPlugin::~ImprovedStitchingPlugin()
{
	DeleteICPObjects();

	delete m_Widget;
}


//----------------------------------------------------------------------------
QString
ImprovedStitchingPlugin::GetName()
{
	return tr("ImprovedStitchingPlugin");
}


//----------------------------------------------------------------------------
QWidget*
ImprovedStitchingPlugin::GetPluginGUI()
{
	return m_Widget;
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::ProcessEvent(ritk::Event::Pointer EventP)
{
	// New frame event
	if ( EventP->type() == ritk::NewFrameEvent::EventType )
	{
		// Cast the event
		ritk::NewFrameEvent::Pointer NewFrameEventP = qSharedPointerDynamicCast<ritk::NewFrameEvent,ritk::Event>(EventP);
		if ( !NewFrameEventP )
		{
			LOG_DEB("Event mismatch detected: Type=" << EventP->type());
			return;
		}
		ritk::NewFrameEvent::RImageConstPointer CurrentFrameP = NewFrameEventP->RImage;

		// We have to lock as this member may be used by another thread
		LockPlugin();
		m_CurrentFrame = CurrentFrameP;
		UnlockPlugin();

		// Stitching
		if (m_Widget->m_CheckBoxAutoStitch->isChecked())
		{
			emit FrameToStitchAvailable(m_Widget->m_CheckBoxVisualizeVolume->isChecked());
		}
	} 
	else
	{
		LOG_DEB("Unknown event");
	}
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::AutoStitch(bool Visualize)
{
	if (m_IsFirstFrame)
	{
		LockPlugin();
		m_KinectMoving->ImportKinectData(m_CurrentFrame);
		UnlockPlugin();

		m_IsFirstFrame = false;
	}
	// Previous moving points are now fixed points
	m_KinectFixed->SwapPointsContainer(m_KinectMoving.GetPointer());

	// In case, the number of landmarks has changed
	m_KinectFixed->SetNumberOfLandmarks(m_NumPts);

	// Extract landmarks from "swapped" point cloud (we can not use the landmarks
	// from the set of moving points, because we do NOT clip the fixed set of points)
	m_KinectFixed->ExtractLandmarks();

	// New moving points
	m_KinectMoving->SetNumberOfLandmarks(m_NumPts);

	LockPlugin();
	m_KinectMoving->ImportKinectData(m_CurrentFrame);
	UnlockPlugin();

	// Pass data to ICP
	switch (m_NumPts)
	{

#define ICP_SET_POINTS2(X) case X: m_ICP##X->SetFixedPts(m_KinectFixed->GetLandmarkContainer()); m_ICP##X->SetMovingPts(m_KinectMoving->GetLandmarkContainer()); break;

		ICP_SET_POINTS2(512)
		ICP_SET_POINTS2(1024)
		ICP_SET_POINTS2(2048)
		ICP_SET_POINTS2(4096)
		ICP_SET_POINTS2(8192)
		ICP_SET_POINTS2(16384)
	}

	// Start ICP
	RunICP();
	
	// Add moving points to volume
	/*float Time = 0;
	cudaEvent_t start_event, stop_event;
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
	cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);*/

	m_Volume->AddPoints(m_KinectMoving->GetPtsContainer());

	/*cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&Time, start_event, stop_event);
	LOG_DEB("AddPoints: " << Time << "ms");*/
	

	// Visualize the current volume, if requested
	if (Visualize)
	{

//#define MEASURE_RUNTIME
#ifdef MEASURE_RUNTIME
		float Time = 0;
		cudaEvent_t start_event, stop_event;
		cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
		cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
		cudaEventRecord(start_event, 0);
#endif

		// Copy data to cudaArray
		dynamic_cast<Cuda3DArrayContainerType*>(m_VolumeForRaycasting->GetPixelContainer())->DeepCopy(m_Volume->GetPixelContainer());
		m_VolumeForRaycasting->Modified();

		m_VolumeRayCastEntity->SetVolume(m_VolumeForRaycasting);

#ifdef MEASURE_RUNTIME
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&Time, start_event, stop_event);
		LOG_DEB("DeepCopy: " << Time << "ms");

#endif
	}	
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::ChangeNumPts()
{
	// Get desired number of points from GUI
	int selected = m_Widget->m_ComboBoxICPNumPts->currentText().toInt();

	m_NumPts = selected;
	DeleteICPObjects();
	switch (m_NumPts) 
	{

#define ICP_CONSTRUCTOR(X) case X: if (!m_ICP##X) m_ICP##X = new ICP<X, ICP_DATA_DIM>; m_ICP##X->SetPayloadWeight(m_PayloadWeight); break;

		ICP_CONSTRUCTOR(512)
		ICP_CONSTRUCTOR(1024)
		ICP_CONSTRUCTOR(2048)
		ICP_CONSTRUCTOR(4096)
		ICP_CONSTRUCTOR(8192)
		ICP_CONSTRUCTOR(16384)
	}

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::DeleteICPObjects()
{
#define ICP_DELETE(X) if (m_ICP##X) { delete m_ICP##X; m_ICP##X = NULL; }
	ICP_DELETE(512)
	ICP_DELETE(1024)
	ICP_DELETE(2048)
	ICP_DELETE(4096)
	ICP_DELETE(8192)
	ICP_DELETE(16384)
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::ChangeRGBWeight(int Value)
{
	// Compute new weight from slider value
	m_PayloadWeight = static_cast<float>(Value) / 
		static_cast<float>(m_Widget->m_HorizontalSliderRGBWeight->maximum()); // Inbetween [0;1]

	// Tell ICP to update its internal weight parameters
	switch (m_NumPts) 
	{

#define ICP_SET_PAYLOAD_WEIGHT(X) case X: m_ICP##X->SetPayloadWeight(m_PayloadWeight); break;

	ICP_SET_PAYLOAD_WEIGHT(512)
	ICP_SET_PAYLOAD_WEIGHT(1024)
	ICP_SET_PAYLOAD_WEIGHT(2048)
	ICP_SET_PAYLOAD_WEIGHT(4096)
	ICP_SET_PAYLOAD_WEIGHT(8192)
	ICP_SET_PAYLOAD_WEIGHT(16384)
	}

	// Update StatusTip and ToolTip
	m_Widget->m_HorizontalSliderRGBWeight->setToolTip("XYZ vs. RGB Weight: " + QString::number(1.f - m_PayloadWeight) + " vs. " + QString::number(m_PayloadWeight));
	m_Widget->m_HorizontalSliderRGBWeight->setStatusTip("XYZ vs. RGB Weight: " + QString::number(1.f - m_PayloadWeight) + " vs. " + QString::number(m_PayloadWeight));

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::RunICP()
{
	// Runtime statistics
	/*float ICPTime = 0;
	cudaEvent_t start_event, stop_event;
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
	cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);*/

	// Here we store number of iterations and final matrix norm
	unsigned long NumIter;
	float FinalNorm;

	// Run ICP and show resulting set of moving points
	switch (m_NumPts) 
	{

#define ICP_RUN(X)	case X: m_ICP##X->Run(&NumIter, &FinalNorm);																		\
	/*cudaEventRecord(stop_event, 0);*/																									\
	/*cudaEventSynchronize(stop_event);*/																								\
	/*cudaEventElapsedTime(&ICPTime, start_event, stop_event);*/																		\
	m_KinectMoving->TransformPts(m_ICP##X->GetTransformationMatrixContainer());															\
	break;	

		ICP_RUN(512)
		ICP_RUN(1024)
		ICP_RUN(2048)
		ICP_RUN(4096)
		ICP_RUN(8192)
		ICP_RUN(16384)
	}

	// Compute and print average runtime
	/*float TimePerIter = ICPTime/static_cast<float>(NumIter);
	LOG_DEB("ICP: " << ICPTime << "ms. Iterations: " << NumIter << " => " << TimePerIter << "ms/iter");*/
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::ChangeClipPercentage(int Value)
{
	// Compute new clip percentage from horizontal slider value, such that ClipPercentage in [0;0.5[
	float ClipPercentage = static_cast<float>(Value) / 
		static_cast<float>(m_Widget->m_HorizontalSliderClipPercentage->maximum()) / 2.1f;

	// Update moving landmarks
	m_KinectMoving->SetClipPercentage(ClipPercentage);

	// Update StatusTip and ToolTip
	m_Widget->m_HorizontalSliderClipPercentage->setToolTip("Clip Percentage: " + QString::number(ClipPercentage));
	m_Widget->m_HorizontalSliderClipPercentage->setStatusTip("Clip Percentage: " + QString::number(ClipPercentage));

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::SaveVolume()
{
	QString FileName = QFileDialog::getSaveFileName(this->m_Widget,
		tr("Save Volume"), "C:/_DATA/volume.mha", tr("MetaImage Files (*.mha)"));

	// We have to make sure, that the voxels are synchronized to the host
	m_Volume->SynchronizeDataToHost();

	if (!FileName.isEmpty())
	{
		// Use ITK for file writing
		typedef itk::ImageFileWriter<VolumeManager> WriterType;
		WriterType::Pointer writer = WriterType::New();
		writer->SetFileName(FileName.toStdString().c_str());
		writer->SetInput(m_Volume);
		writer->Update();

		// Compute filesize
		std::ifstream InStream(FileName.toStdString().c_str());
		InStream.seekg(0, std::ios::end);
		std::ios::pos_type pos = InStream.tellg();
		float FileSize = static_cast<float>(pos)/1024.f/1024.f;

		LOG_DEB("Volume saved: " << FileName.toStdString().c_str() << " (" << FileSize << " MiB)");
	}
}


//----------------------------------------------------------------------------
void
ImprovedStitchingPlugin::ResetVolume()
{
	// Set all voxels to 0
	m_Volume->ResetVolume();

	// Also, reset the cudaArray used for raycasting
	dynamic_cast<Cuda3DArrayContainerType*>(m_VolumeForRaycasting->GetPixelContainer())->DeepCopy(m_Volume->GetPixelContainer());
	m_VolumeForRaycasting->Modified();

	m_VolumeRayCastEntity->SetVolume(m_VolumeForRaycasting);

	// Call this method to reset the ICP also, since the ICP stores a initial
	// transformation (which is the accumulation of all transformations so far)
	ChangeNumPts();

	m_IsFirstFrame = true;
}

