#include "FastICPPlugin.h"
#include "defs.h"

#include "ritkDebugManager.h"
#include "ritkManager.h"
#include "ritkRGBRImage.h"

#include "vtkMath.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkLine.h"
#include "vtkUnsignedCharArray.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkProperty.h"
#include "vtkRendererCollection.h"

#include "itkImageFileWriter.h"

#include <QFileDialog>

// Required for Sleep() in animated ICP
#ifdef WIN32
#include "windows.h"
#endif


//----------------------------------------------------------------------------
FastICPPlugin::FastICPPlugin()
{
	// Create the widget
	m_Widget = new FastICPWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget, SLOT(UpdateGUI()));

	// -- Plugin-specific signals/slots --
	// ICP-related
	connect(m_Widget->m_PushButtonRunICP, SIGNAL(clicked()), this, SLOT(RunICP()));
	connect(m_Widget->m_ComboBoxICPNumPts, SIGNAL(currentIndexChanged(int)), this, SLOT(ChangeNumPts()));
	connect(m_Widget->m_CheckBoxPlotDistances, SIGNAL(stateChanged(int)), this, SLOT(PlotCorrespondenceLinesWrapper()));
	connect(m_Widget->m_HorizontalSliderRGBWeight, SIGNAL(valueChanged(int)), this, SLOT(ChangeRGBWeight(int)));
	connect(m_Widget->m_RadioButtonSyntheticData, SIGNAL(toggled(bool)), this, SLOT(ToggleDataMode()));

	// Data-related: synthetic
	connect(m_Widget->m_PushButtonGenerateSampleData, SIGNAL(clicked()), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_ComboBoxDataDistribution, SIGNAL(currentIndexChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_CheckBoxLUTColor, SIGNAL(stateChanged(int)), this, SLOT(GenerateSyntheticData()));	
	connect(m_Widget->m_HorizontalSliderRotX, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_HorizontalSliderRotY, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_HorizontalSliderRotZ, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_HorizontalSliderTransX, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_HorizontalSliderTransY, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_HorizontalSliderTransZ, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));
	connect(m_Widget->m_HorizontalSliderNoisyData, SIGNAL(valueChanged(int)), this, SLOT(GenerateSyntheticData()));

	// Data-related: real (Kinect)
	connect(m_Widget->m_PushButtonImportFrameFixed, SIGNAL(clicked()), this, SLOT(ImportFixedData()));
	connect(m_Widget->m_PushButtonImportFrameMoving, SIGNAL(clicked()), this, SLOT(ImportMovingData()));
	connect(m_Widget->m_HorizontalSliderClipPercentage, SIGNAL(valueChanged(int)), this, SLOT(ChangeClipPercentage(int)));
	connect(m_Widget->m_CheckBoxShowLandmarks, SIGNAL(stateChanged(int)), this, SLOT(ToggleShowLandmarks()));

	
	// Init members
	m_ActorFixed = ActorPointer::New();
	m_ActorFixed->SetVisualizationMode(ritk::RImageActorPipeline::RGB);
	m_ActorMoving = ActorPointer::New();
	m_ActorMoving->SetVisualizationMode(ritk::RImageActorPipeline::RGB);
	m_ActorLines = ActorPointer::New();
	m_ActorLines->SetVisualizationMode(ritk::RImageActorPipeline::RGB);

	// Improves visualization
	m_ActorFixed->GetProperty()->SetPointSize(2);
	m_ActorMoving->GetProperty()->SetPointSize(2);

	// Add actors to visualization unit
	m_Widget->m_VisualizationWidget3D->AddActor(m_ActorFixed);
	m_Widget->m_VisualizationWidget3D->AddActor(m_ActorMoving);
	m_Widget->m_VisualizationWidget3D->AddActor(m_ActorLines);

	// Set Background to black
	m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->SetBackground(.1, .1, .1);
	m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->SetBackground2(.3, .3, .3);

	// Init ICP objects to NULL
	m_ICP512   = NULL;
	m_ICP1024  = NULL;
	m_ICP2048  = NULL;
	m_ICP4096  = NULL;
	m_ICP8192  = NULL;
	m_ICP16384 = NULL;

	// This will set m_NumPts and init the appropriate ICP object
	ChangeNumPts();

	// Init RGB weight
	ChangeRGBWeight(m_Widget->m_HorizontalSliderRGBWeight->value());

	// Init data generator
	m_DataGenerator = DataGenerator::New();

	// Init member variable that stores current plugin mode (synthetic or Kinect data)
	m_SyntheticDataMode = m_Widget->m_RadioButtonSyntheticData->isChecked();

	// Init switch for landmarks vs all points
	m_ShowLandmarks = m_Widget->m_CheckBoxShowLandmarks->isChecked();

	// Init KinectDataManagers
	m_KinectFixed = KinectDataManager::New();
	m_KinectMoving = KinectDataManager::New();

	// Init clip percentage for Kinect data manager (moving points)
	float ClipPercentage = static_cast<float>(m_Widget->m_HorizontalSliderClipPercentage->value()) / 
		static_cast<float>(m_Widget->m_HorizontalSliderClipPercentage->maximum()) / 2.01f; // Roughly inbetween [0;0.5[
	m_KinectMoving->SetClipPercentage(ClipPercentage);
	m_KinectFixed->SetClipPercentage(0);

	// Init flags
	m_KinectFixedImported = false;
	m_KinectMovingImported = false;
	m_FrameAvailable = false;
}


//----------------------------------------------------------------------------
FastICPPlugin::~FastICPPlugin()
{
	DeleteICPObjects();

	delete m_Widget;
}


//----------------------------------------------------------------------------
QString
FastICPPlugin::GetName()
{
	return tr("FastICPPlugin");
}


//----------------------------------------------------------------------------
QWidget*
FastICPPlugin::GetPluginGUI()
{
	return m_Widget;
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ProcessEvent(ritk::Event::Pointer EventP)
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

		m_FrameAvailable = true;
	} 
	else
	{
		LOG_DEB("Unknown event");
	}
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ChangeNumPts()
{
	// Get desired number of points from GUI
	int selected = m_Widget->m_ComboBoxICPNumPts->currentText().toInt();

	// Delete previous and create new ICP object if appropriate
	if (m_NumPts != selected)
	{
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
	}

	// Reset flags
	m_KinectFixedImported = false;
	m_KinectMovingImported = false;

	// Clear preview window (all actors)
	vtkSmartPointer<vtkPolyData> EmptyPolyData = vtkSmartPointer<vtkPolyData>::New();
	m_ActorFixed->SetData(EmptyPolyData, true);
	m_ActorMoving->SetData(EmptyPolyData, true);
	m_ActorLines->SetData(EmptyPolyData, true);

	// Disable ICP button
	m_Widget->m_PushButtonRunICP->setDisabled(true);

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::DeleteICPObjects()
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
FastICPPlugin::CopyFloatDataToActor(float* Data, unsigned long NumPts, ActorPointer Actor)
{
	// Init all required VTK data structures
	vtkSmartPointer<vtkPoints> Points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> Cells = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPolyData> PolyData = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkUnsignedCharArray> Colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
	Colors->SetNumberOfComponents(4);

	// Use first 3 dimensions as XYZ coordinates, 4'th to 6'th dimension as RGB payload
	for (int i = 0; i < NumPts; ++i)
	{
		Points->InsertNextPoint(Data[i*ICP_DATA_DIM+0], Data[i*ICP_DATA_DIM+1], Data[i*ICP_DATA_DIM+2]);
		Colors->InsertNextTuple4(Data[i*ICP_DATA_DIM+3], Data[i*ICP_DATA_DIM+4], Data[i*ICP_DATA_DIM+5], 220);
	}

	for (vtkIdType i = 0; i < Points->GetNumberOfPoints(); i++)
	{
		Cells->InsertNextCell(1, &i);
	}	

	// Create polydata
	PolyData->SetPoints(Points);
	PolyData->GetPointData()->SetScalars(Colors);
	PolyData->SetVerts(Cells);
	PolyData->Update();

	/*double bounds[6];
	PolyData->ComputeBounds();
	PolyData->GetBounds(bounds);
	std::stringstream boundsStream;
	std::copy(bounds, bounds+6, std::ostream_iterator<double>(boundsStream, " "));
	LOG_DEB("bounds: " << boundsStream.str());*/

	// Update actor
	Actor->SetData(PolyData, false);
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ChangeRGBWeight(int Value)
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
FastICPPlugin::RunICP()
{
	if ( !m_Widget->m_CheckBoxAnimate->isChecked() )
	{
		// Runtime statistics
		float ICPTime = 0;
		cudaEvent_t start_event, stop_event;
		cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
		cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
		cudaEventRecord(start_event, 0);

		// Here we store number of iterations and final matrix norm
		unsigned long NumIter;
		float FinalNorm;

		// Run ICP and show resulting set of moving points
		switch (m_NumPts) 
		{

#define ICP_RUN(X)	case X: m_ICP##X->Run(&NumIter, &FinalNorm);																								\
							cudaEventRecord(stop_event, 0);																										\
							cudaEventSynchronize(stop_event);																									\
							cudaEventElapsedTime(&ICPTime, start_event, stop_event);																			\
						if ( !m_SyntheticDataMode )																												\
							m_KinectMoving->TransformPts(m_ICP##X->GetTransformationMatrixContainer());															\
						if ( m_SyntheticDataMode || ( !m_SyntheticDataMode && m_ShowLandmarks ) )																\
							CopyFloatDataToActor(m_ICP##X->GetMovingPts(), m_NumPts, m_ActorMoving);															\
						if ( !m_SyntheticDataMode && !m_ShowLandmarks )																							\
							CopyFloatDataToActor(m_KinectMoving->GetPtsContainer()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, m_ActorMoving);	\
						break;	
	
		ICP_RUN(512)
		ICP_RUN(1024)
		ICP_RUN(2048)
		ICP_RUN(4096)
		ICP_RUN(8192)
		ICP_RUN(16384)
		}

		// Compute and print average runtime
		float TimePerIter = ICPTime/static_cast<float>(NumIter);
		LOG_DEB("ICP: " << ICPTime << "ms. Iterations: " << NumIter << " => " << TimePerIter << "ms/iter");

		// Plot lines
		PlotCorrespondenceLinesWrapper();

		emit UpdateGUI();
	}
	else
	{
		if ( !m_SyntheticDataMode && !m_Widget->m_CheckBoxShowLandmarks->isChecked() )
		{
			LOG_DEB("Animation for entire Kinect image is not implemented, try landmarks only or synthetic data");
			return;
		}

		m_Widget->m_GroupBoxControls->setEnabled(false);

		// Initialize ICP first
		switch (m_NumPts) 
		{

#define ICP_INITIALIZE(X)	case X: m_ICP##X->Initialize();	break;

				ICP_INITIALIZE(512)
				ICP_INITIALIZE(1024)
				ICP_INITIALIZE(2048)
				ICP_INITIALIZE(4096)
				ICP_INITIALIZE(8192)
				ICP_INITIALIZE(16384)
		}

		// Iterate until convergence
		bool AnotherIteration = true;
		while (AnotherIteration)
		{
			#ifdef WIN32
			QTime Timer;
			Timer.start();
			#endif

			switch (m_NumPts)
			{

#define ICP_ITERATE(X)	case X: AnotherIteration = m_ICP##X->NextIteration();																							\
							if ( !AnotherIteration && !m_SyntheticDataMode )																							\
								m_KinectMoving->TransformPts(m_ICP##X->GetTransformationMatrixContainer());																\
							if ( m_SyntheticDataMode || ( !m_SyntheticDataMode && m_ShowLandmarks ) )																	\
								m_ICP##X->GetMovingPtsContainer()->SynchronizeHost(); CopyFloatDataToActor(m_ICP##X->GetMovingPts(), m_NumPts, m_ActorMoving);			\
							if ( !m_SyntheticDataMode && !m_ShowLandmarks )																								\
								CopyFloatDataToActor(m_KinectMoving->GetPtsContainer()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, m_ActorMoving);	\
							break;	

			ICP_ITERATE(512)
			ICP_ITERATE(1024)
			ICP_ITERATE(2048)
			ICP_ITERATE(4096)
			ICP_ITERATE(8192)
			ICP_ITERATE(16384)
			}

			// Plot lines
			PlotCorrespondenceLinesWrapper();

			emit UpdateGUI();

			// Force Qt event processing now (required for animation effect)
			QApplication::processEvents();

			#ifdef WIN32
			bool SlowMotion = m_Widget->m_CheckBoxAnimateICPSlowMotion->isChecked();
			int MaxSleepTime = SlowMotion ? 1000/15 : 1000/60; // maximum of 15 fps or 60 fps respectively
			int TimeLeftToSleep = MaxSleepTime - Timer.elapsed();
			if (TimeLeftToSleep > 0)
				Sleep(TimeLeftToSleep);
			#endif
		}

		m_Widget->m_GroupBoxControls->setEnabled(true);
	}
}


//----------------------------------------------------------------------------
void
FastICPPlugin::GenerateSyntheticData()
{
	// Use DataGenerator to generate data and visualize fixed and moving points
	const float RotationFactor = 180.f; // maximum: 180 degrees
	const float TranslationFactor = 512.f; // no specific meaning
	const float NoiseStdDevFactor = 10.f; // no specific meaning

	// Get parameters from GUI
	float RotX = static_cast<float>(m_Widget->m_HorizontalSliderRotX->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderRotX->maximum()) *
		RotationFactor; 
	float RotY = static_cast<float>(m_Widget->m_HorizontalSliderRotY->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderRotY->maximum()) *
		RotationFactor; 
	float RotZ = static_cast<float>(m_Widget->m_HorizontalSliderRotZ->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderRotZ->maximum()) *
		RotationFactor; 
	float TransX = static_cast<float>(m_Widget->m_HorizontalSliderTransX->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderTransX->maximum()) *
		TranslationFactor;
	float TransY = static_cast<float>(m_Widget->m_HorizontalSliderTransY->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderTransY->maximum()) *
		TranslationFactor;
	float TransZ = static_cast<float>(m_Widget->m_HorizontalSliderTransZ->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderTransZ->maximum()) *
		TranslationFactor;
	float NoisyDataStdDev = static_cast<float>(m_Widget->m_HorizontalSliderNoisyData->sliderPosition()) /
		static_cast<float>(m_Widget->m_HorizontalSliderTransZ->maximum()) *
		NoiseStdDevFactor;

	int DistributionType = m_Widget->m_ComboBoxDataDistribution->currentIndex();
	bool LUTColor = m_Widget->m_CheckBoxLUTColor->isChecked();

	// Pass parameters to DataGenerator
	m_DataGenerator->SetNumberOfPoints(m_NumPts);
	m_DataGenerator->SetTransformationParameters(RotX, RotY, RotZ, TransX, TransY, TransZ);
	m_DataGenerator->SetDistributionType( (DataGenerator::DistributionType) DistributionType );
	m_DataGenerator->SetNoise(NoisyDataStdDev);
	m_DataGenerator->SetColoring(LUTColor);

	// Generate the datasets
	m_DataGenerator->GenerateData();

	// Visualize data
	CopyFloatDataToActor(m_DataGenerator->GetFixedPtsContainer()->GetBufferPointer(), m_NumPts, m_ActorFixed);
	CopyFloatDataToActor(m_DataGenerator->GetMovingPtsContainer()->GetBufferPointer(), m_NumPts, m_ActorMoving);

	// Pass data to ICP
	switch (m_NumPts)
	{

#define ICP_SET_POINTS(X) case X: m_ICP##X->SetFixedPts(m_DataGenerator->GetFixedPtsContainer()); m_ICP##X->SetMovingPts(m_DataGenerator->GetMovingPtsContainer()); break;
		
		ICP_SET_POINTS(512)
		ICP_SET_POINTS(1024)
		ICP_SET_POINTS(2048)
		ICP_SET_POINTS(4096)
		ICP_SET_POINTS(8192)
		ICP_SET_POINTS(16384)
	}
	

	// Enable RunICP button
	m_Widget->m_PushButtonRunICP->setEnabled(true);

	// Reset camera
	if (m_Widget->m_CheckBoxAutoResetCamera->isChecked())
		m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->ResetCamera();
	
	// Plot correspondence lines if wanted
	PlotCorrespondenceLinesWrapper();

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::PlotCorrespondenceLines(float* Data1, float* Data2, ActorPointer Actor)
{
	// Check if we want those lines
	if (!m_Widget->m_CheckBoxPlotDistances->isChecked())
	{
		m_Widget->m_VisualizationWidget3D->RemoveActor(m_ActorLines);
		return;
	}

	// Init all required VTK data structures
	vtkSmartPointer<vtkPoints> Points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPolyData> PolyData = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkUnsignedCharArray> Colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
	vtkSmartPointer<vtkCellArray> Lines = vtkSmartPointer<vtkCellArray>::New();
	Colors->SetNumberOfComponents(4);

	// Init variables
	double MinDist = DBL_MAX;
	double MaxDist = DBL_MIN;
	double TotalDist = 0;
	double* Dists = new double[m_NumPts];

	// Use first 3 dimensions as XYZ coordinates, 4'th to 6'th dimension as RGB payload
	for (int i = 0; i < m_NumPts; ++i)
	{
		const float p1[3] = {Data1[i*ICP_DATA_DIM+0], Data1[i*ICP_DATA_DIM+1], Data1[i*ICP_DATA_DIM+2]};
		const float p2[3] = {Data2[i*ICP_DATA_DIM+0], Data2[i*ICP_DATA_DIM+1], Data2[i*ICP_DATA_DIM+2]};
		double Dist = sqrt(vtkMath::Distance2BetweenPoints(reinterpret_cast<const float*>(p1), reinterpret_cast<const float*>(p2)));

		TotalDist += Dist;
		Dists[i] = Dist;

		// Compute minimum and maximum distance
		MinDist = Dist < MinDist ? Dist : MinDist;
		MaxDist = Dist > MaxDist ? Dist : MaxDist;

		Points->InsertNextPoint(p1);
		Points->InsertNextPoint(p2);

		vtkSmartPointer<vtkLine> Line =
			vtkSmartPointer<vtkLine>::New();
		Line->GetPointIds()->SetId(0, 2*i);
		Line->GetPointIds()->SetId(1, 2*i+1);
		Lines->InsertNextCell(Line);
	}

	// Choose color based on relative distance
	double DistRange = MaxDist - MinDist;
	for (int i = 0; i < m_NumPts; ++i)
	{
		const double Val = 55 + ((Dists[i] - MinDist) / DistRange) * 200;
		Colors->InsertNextTuple4(Val, Val, Val, 100);
	}

	delete [] Dists;

	// Create polydata
	PolyData->SetPoints(Points);
	PolyData->GetCellData()->SetScalars(Colors);
	PolyData->SetLines(Lines);
	PolyData->Update();

	// Update actor
	Actor->SetData(PolyData, false);

	// Add actor to visualization widget
	m_Widget->m_VisualizationWidget3D->AddActor(m_ActorLines);
}


//----------------------------------------------------------------------------
void
FastICPPlugin::PlotCorrespondenceLinesWrapper()
{
	// Lines only valid for synthetic data
	if (!m_SyntheticDataMode) return;
	
#define ICP_PLOT_CORRESPONDING_LINES(X) case X: if (m_ICP##X->GetFixedPtsContainer().IsNotNull()) PlotCorrespondenceLines(m_ICP##X->GetFixedPts(), m_ICP##X->GetMovingPts(), m_ActorLines); break;
	switch (m_NumPts) 
	{
	ICP_PLOT_CORRESPONDING_LINES(512)
	ICP_PLOT_CORRESPONDING_LINES(1024)
	ICP_PLOT_CORRESPONDING_LINES(2048)
	ICP_PLOT_CORRESPONDING_LINES(4096)
	ICP_PLOT_CORRESPONDING_LINES(8192)
	ICP_PLOT_CORRESPONDING_LINES(16384)
	}

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ImportFixedData()
{
	if (!m_FrameAvailable)
	{
		LOG_DEB("No frame available!")
		return;
	}

	// Resize KinectDataManager
	m_KinectFixed->SetNumberOfLandmarks(m_NumPts);

	// Pass current frame to KinectDataManager
	LockPlugin();
	m_KinectFixed->ImportKinectData(m_CurrentFrame);
	UnlockPlugin();

	// Visualize fixed points
	if (m_ShowLandmarks)
		CopyFloatDataToActor(m_KinectFixed->GetLandmarkContainer()->GetBufferPointer(), m_NumPts, m_ActorFixed);
	else
		CopyFloatDataToActor(m_KinectFixed->GetPtsContainer()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, m_ActorFixed);

	// Pass extracted landmarks to ICP
	switch (m_NumPts)
	{

#define ICP_SET_POINTS_KINECT_FIXED(X) case X: m_ICP##X->SetFixedPts(m_KinectFixed->GetLandmarkContainer()); break;
		
		ICP_SET_POINTS_KINECT_FIXED(512)
		ICP_SET_POINTS_KINECT_FIXED(1024)
		ICP_SET_POINTS_KINECT_FIXED(2048)
		ICP_SET_POINTS_KINECT_FIXED(4096)
		ICP_SET_POINTS_KINECT_FIXED(8192)
		ICP_SET_POINTS_KINECT_FIXED(16384)
	}

	// Reset camera
	if (m_Widget->m_CheckBoxAutoResetCamera->isChecked())
		m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->ResetCamera();

	// Update member
	m_KinectFixedImported = true;

	// Enable ICP button
	if (m_KinectFixedImported && m_KinectMovingImported)
		m_Widget->m_PushButtonRunICP->setEnabled(true);

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ImportMovingData()
{
	if (!m_FrameAvailable)
	{
		std::cout << "No frame available!" << std::endl;
		return;
	}

	// Resize KinectDataManager
	m_KinectMoving->SetNumberOfLandmarks(m_NumPts);

	// Pass current frame to KinectDataManager
	LockPlugin();
	m_KinectMoving->ImportKinectData(m_CurrentFrame);
	UnlockPlugin();

	// Visualize moving points
	if (m_ShowLandmarks)
		CopyFloatDataToActor(m_KinectMoving->GetLandmarkContainer()->GetBufferPointer(), m_NumPts, m_ActorMoving);
	else
		CopyFloatDataToActor(m_KinectMoving->GetPtsContainer()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, m_ActorMoving);

	// Pass extracted landmarks to ICP
	switch (m_NumPts)
	{

#define ICP_SET_POINTS_KINECT_MOVING(X) case X: m_ICP##X->SetMovingPts(m_KinectMoving->GetLandmarkContainer()); break;
		
		ICP_SET_POINTS_KINECT_MOVING(512)
		ICP_SET_POINTS_KINECT_MOVING(1024)
		ICP_SET_POINTS_KINECT_MOVING(2048)
		ICP_SET_POINTS_KINECT_MOVING(4096)
		ICP_SET_POINTS_KINECT_MOVING(8192)
		ICP_SET_POINTS_KINECT_MOVING(16384)
	}

	// Reset camera
	if (m_Widget->m_CheckBoxAutoResetCamera->isChecked())
		m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->ResetCamera();

	// Update member
	m_KinectMovingImported = true;

	// Enable ICP button
	if (m_KinectFixedImported && m_KinectMovingImported)
		m_Widget->m_PushButtonRunICP->setEnabled(true);

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ChangeClipPercentage(int Value)
{
	// Compute new clip percentage from horizontal slider value, such that ClipPercentage in [0;0.5[
	float ClipPercentage = static_cast<float>(Value) / 
		static_cast<float>(m_Widget->m_HorizontalSliderClipPercentage->maximum()) / 2.1f;

	// Update moving landmarks
	m_KinectMoving->SetClipPercentage(ClipPercentage);

	// Visualize landmarks, if appropriate
	if (m_ShowLandmarks)
		CopyFloatDataToActor(m_KinectMoving->GetLandmarkContainer()->GetBufferPointer(), m_NumPts, m_ActorMoving);

	// Update StatusTip and ToolTip
	m_Widget->m_HorizontalSliderClipPercentage->setToolTip("Clip Percentage: " + QString::number(ClipPercentage));
	m_Widget->m_HorizontalSliderClipPercentage->setStatusTip("Clip Percentage: " + QString::number(ClipPercentage));

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ToggleDataMode()
{
	// Update member
	m_SyntheticDataMode = m_Widget->m_RadioButtonSyntheticData->isChecked();

	// Clear preview window (all actors)
	vtkSmartPointer<vtkPolyData> EmptyPolyData = vtkSmartPointer<vtkPolyData>::New();
	m_ActorFixed->SetData(EmptyPolyData, true);
	m_ActorMoving->SetData(EmptyPolyData, true);
	m_ActorLines->SetData(EmptyPolyData, true);

	emit UpdateGUI();
}


//----------------------------------------------------------------------------
void
FastICPPlugin::ToggleShowLandmarks()
{
	// Update member
	m_ShowLandmarks = m_Widget->m_CheckBoxShowLandmarks->isChecked();

	// Visualize
	if (m_ShowLandmarks)
	{
		CopyFloatDataToActor(m_KinectFixed->GetLandmarkContainer()->GetBufferPointer(), m_NumPts, m_ActorFixed);
		CopyFloatDataToActor(m_KinectMoving->GetLandmarkContainer()->GetBufferPointer(), m_NumPts, m_ActorMoving);
	} else
	{
		CopyFloatDataToActor(m_KinectFixed->GetPtsContainer()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, m_ActorFixed);
		CopyFloatDataToActor(m_KinectMoving->GetPtsContainer()->GetBufferPointer(), KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, m_ActorMoving);
	}

	emit UpdateGUI();
}
