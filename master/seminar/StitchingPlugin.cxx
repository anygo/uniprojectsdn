// standard includes
#include "StitchingPlugin.h"
#include "DebugManager.h"
#include "Manager.h"

// Qt includes
#include <QFileDialog.h>
#include <QTime>

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

// our includes
#include <ExtendedICPTransform.h>
#include <ClosestPointFinder.h>
#include <ClosestPointFinderBruteForceCPU.h>



StitchingPlugin::StitchingPlugin()
{
	// create the widget
	m_Widget = new StitchingWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget,							SLOT(UpdateGUI()));
	connect(this, SIGNAL(UpdateGUI()), m_Widget->m_VisualizationWidget3D,	SLOT(UpdateGUI()));

	// set signals (from buttons, etc) and slots (in this file)
	connect(m_Widget->m_PushButtonLoadFrame,			SIGNAL(clicked()),			this, SLOT(LoadFrame()));
	connect(m_Widget->m_PushButtonCleanFrame,			SIGNAL(clicked()),			this, SLOT(CleanFrame()));
	connect(m_Widget->m_PushButtonLoadCleanStitch,		SIGNAL(clicked()),			this, SLOT(LoadCleanStitch()));
	connect(m_Widget->m_PushButtonCleanWorld,			SIGNAL(clicked()),			this, SLOT(CleanWorld()));
	connect(m_Widget->m_PushButtonDelaunay2D,			SIGNAL(clicked()),			this, SLOT(Delaunay2D()));
	connect(m_Widget->m_PushButtonSaveVTKData,			SIGNAL(clicked()),			this, SLOT(SaveVTKData()));
	connect(m_Widget->m_PushButtonInitializeWorld,		SIGNAL(clicked()),			this, SLOT(InitializeWorld()));
	connect(m_Widget->m_PushButtonStitchToWorld,		SIGNAL(clicked()),			this, SLOT(StitchToWorld()));
	connect(m_Widget->m_HorizontalSliderPointSize,		SIGNAL(valueChanged(int)),	this, SLOT(ChangeVisualizationProperties()));
		
	// add data actor
	m_DataActor3D = vtkSmartPointer<ritk::RImageActorPipeline>::New();	
	m_DataActor3D->SetVisualizationMode(ritk::RImageActorPipeline::RGB);
	m_Widget->m_VisualizationWidget3D->AddActor(m_DataActor3D);

	// initialize member objects
	m_Data =					vtkSmartPointer<vtkPolyData>::New();
	m_TheWorld =				vtkSmartPointer<vtkPolyData>::New();
	m_PreviousFrame =			vtkSmartPointer<vtkPolyData>::New();
	m_PreviousTransformMatrix =	vtkSmartPointer<vtkMatrix4x4>::New();
}
StitchingPlugin::~StitchingPlugin()
{
	delete m_Widget;
}
//----------------------------------------------------------------------------
QString
StitchingPlugin::GetName()
{
	return tr("StitchingPlugin");
}
//----------------------------------------------------------------------------
QWidget*
StitchingPlugin::GetPluginGUI()
{
	return m_Widget;
}
//----------------------------------------------------------------------------
void
StitchingPlugin::ChangeVisualizationProperties()
{
	m_DataActor3D->GetProperty()->SetPointSize(m_Widget->m_HorizontalSliderPointSize->value());
	//m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->SetBackground(0.2, 0.3, 0.9);

	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::ProcessEvent(ritk::Event::Pointer EventP)
{
	// New frame event
	if ( EventP->type() == ritk::NewFrameEvent::EventType )
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
		if (m_Widget->m_CheckBoxAutoStitch->isChecked())
		{
			LoadCleanStitch();
		}

		// enable buttons (ProcessEvent has to be called at least once before
		// we can load the data into our plugin)
		m_Widget->m_PushButtonLoadFrame->setEnabled(true);
		
		emit UpdateGUI();
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
//----------------------------------------------------------------------------
void
StitchingPlugin::LoadCleanStitch()
{
	QTime t = QTime::currentTime();

	t.start();
	LoadFrame(false);
	std::cout << "LoadFrame():     " << t.elapsed() << " ms" << std::endl;

	t.start();
	CleanFrame(false);
	std::cout << "CleanFrame():    " << t.elapsed() << " ms" << std::endl;

	t.start();
	StitchToWorld(false);
	std::cout << "StitchToWorld(): " << t.elapsed() << " ms" << std::endl;


	m_DataActor3D->SetData(m_TheWorld, false);

	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::LoadFrame(bool update)
{
	m_DataActor3D->SetData(m_CurrentFrame);
	m_Data->ShallowCopy(m_DataActor3D->GetData());

	// remove invalid points
	ExtractValidPoints();

	if (update)
	{
		// enable buttons
		m_Widget->m_PushButtonCleanFrame->setEnabled(true);
		m_Widget->m_PushButtonInitializeWorld->setEnabled(true);

		emit UpdateGUI();
		m_Widget->m_VisualizationWidget3D->UpdateGUI();
	}
}
//----------------------------------------------------------------------------
void
StitchingPlugin::ExtractValidPoints()
{
	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> cells =
		vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkDataArray> colors =
		vtkSmartPointer<vtkUnsignedCharArray>::New();
		
	colors->SetNumberOfComponents(4);


	double p[3];
	for (vtkIdType i = 0; i < m_Data->GetNumberOfPoints(); ++i)
	{	
		m_Data->GetPoint(i, p);

		if (p[0] == p[0]) // i.e. not QNAN
		{
			points->InsertNextPoint(p[0], p[1], p[2]);
			colors->InsertNextTuple(m_Data->GetPointData()->GetScalars()->GetTuple(i));
		}
	}

	for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
	{
		cells->InsertNextCell(1, &i);
	}	

	// update m_Data
	m_Data->SetPoints(points);
	m_Data->GetPointData()->SetScalars(colors);
	m_Data->SetVerts(cells);
	m_Data->Update();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::Clip(vtkPolyData *toBeClipped)
{
	vtkSmartPointer<vtkClipPolyData> clipper =
		vtkSmartPointer<vtkClipPolyData>::New();	
	vtkSmartPointer<vtkBox> box =
		vtkSmartPointer<vtkBox>::New();

	double bounds[6];
	toBeClipped->GetBounds(bounds);

	// modify x, y (and z) bounds
	bounds[0] = bounds[0] + m_Widget->m_DoubleSpinBoxClipPercentage->value()*(bounds[1] - bounds[0]);
	bounds[1] = bounds[1] - m_Widget->m_DoubleSpinBoxClipPercentage->value()*(bounds[1] - bounds[0]);
	bounds[2] = bounds[2] + m_Widget->m_DoubleSpinBoxClipPercentage->value()*(bounds[3] - bounds[2]);
	bounds[3] = bounds[3] - m_Widget->m_DoubleSpinBoxClipPercentage->value()*(bounds[3] - bounds[2]);
	bounds[4] = bounds[4] + m_Widget->m_DoubleSpinBoxClipPercentage->value()*(bounds[5] - bounds[4]);
	bounds[5] = bounds[5] - m_Widget->m_DoubleSpinBoxClipPercentage->value()*(bounds[5] - bounds[4]);

	box->SetBounds(bounds);
	clipper->SetClipFunction(box);
	clipper->InsideOutOn();
	clipper->SetInput(toBeClipped);
	clipper->Update();

	toBeClipped->ShallowCopy(clipper->GetOutput());
}
//----------------------------------------------------------------------------
void
StitchingPlugin::CleanFrame(bool update)
{
	Clean(m_Data);

	if (update)
	{
		m_DataActor3D->SetData(m_Data, false);

		emit UpdateGUI();
	}
}
void
StitchingPlugin::CleanWorld()
{
	Clean(m_TheWorld);

	m_DataActor3D->SetData(m_TheWorld, false);

	// update number of points label
	m_Widget->m_lcdNumberPointsInWorld->display(m_TheWorld->GetNumberOfPoints());

	emit UpdateGUI();
}
void
StitchingPlugin::Clean(vtkPolyData *toBeCleaned)
{
	vtkSmartPointer<vtkCleanPolyData> cleanPolyData =
		vtkSmartPointer<vtkCleanPolyData>::New();
	cleanPolyData->SetTolerance(m_Widget->m_DoubleSpinBoxCleanTolerance->value());
	cleanPolyData->PointMergingOn();
	cleanPolyData->ConvertLinesToPointsOn();
	cleanPolyData->ConvertPolysToLinesOn();
	cleanPolyData->ConvertStripsToPolysOn();
	cleanPolyData->SetInput(toBeCleaned);
	cleanPolyData->Update();

	toBeCleaned->ShallowCopy(cleanPolyData->GetOutput());
}
//----------------------------------------------------------------------------
void
StitchingPlugin::Delaunay2D()
{
	vtkSmartPointer<vtkTransform> t = 
		vtkSmartPointer<vtkTransform>::New();
	vtkSmartPointer<vtkDelaunay2D> Delaunay2D = 
		vtkSmartPointer<vtkDelaunay2D>::New();
	Delaunay2D->SetInput(m_TheWorld);
	Delaunay2D->SetTransform(t);
	Delaunay2D->Update();
	m_TheWorld->DeepCopy(Delaunay2D->GetOutput());

	m_DataActor3D->SetData(m_TheWorld, false);

	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::InitializeWorld()
{
	// copy data
	m_TheWorld->DeepCopy(m_Data);
	m_PreviousFrame->DeepCopy(m_Data);
	m_PreviousTransformMatrix->Identity();

	// enable buttons
	m_Widget->m_PushButtonStitchToWorld->setEnabled(true);
	m_Widget->m_PushButtonCleanWorld->setEnabled(true);
	m_Widget->m_PushButtonDelaunay2D->setEnabled(true);
	m_Widget->m_PushButtonLoadCleanStitch->setEnabled(true);
	m_Widget->m_CheckBoxAutoStitch->setEnabled(true);

	m_DataActor3D->SetData(m_TheWorld, false);

	// update number of points label
	m_Widget->m_lcdNumberPointsInWorld->display(m_TheWorld->GetNumberOfPoints());

	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::StitchToWorld(bool update)
{
	// iterative closest point (ICP) transformation
	vtkSmartPointer<ExtendedICPTransform> icp = 
		vtkSmartPointer<ExtendedICPTransform>::New();

	if (m_Widget->m_CheckBoxUsePreviousTransformation->isChecked())
	{
		// start with previous transform
		vtkSmartPointer<vtkTransform> prevTrans =
			vtkSmartPointer<vtkTransform>::New();
		prevTrans->SetMatrix(m_PreviousTransformMatrix);
		prevTrans->Modified();

		vtkSmartPointer<vtkTransformPolyDataFilter> previousTransformFilter =
			vtkSmartPointer<vtkTransformPolyDataFilter>::New();

		previousTransformFilter->SetInput(m_Data);
		previousTransformFilter->SetTransform(prevTrans);
		previousTransformFilter->Modified();
		previousTransformFilter->Update();
		m_Data->DeepCopy(previousTransformFilter->GetOutput());
	}

	// get a subvolume of the original data
	vtkSmartPointer<vtkPolyData> voi = 
		vtkSmartPointer<vtkPolyData>::New();
	voi->DeepCopy(m_Data);
	Clip(voi);

	// initialize ClosestPointFinder
	ClosestPointFinder* cpf = new ClosestPointFinderBruteForceCPU(m_Widget->m_SpinBoxMaxLandmarks->value());
	cpf->SetUseRGBData(m_Widget->m_CheckBoxUseRGBData->isChecked());
	cpf->SetWeightRGB(m_Widget->m_DoubleSpinBoxRGBWeight->value());

	// configure icp
	icp->SetSource(voi);
	icp->SetTarget(m_PreviousFrame);
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaxMeanDist(m_Widget->m_DoubleSpinBoxMaxRMS->value());
	icp->SetNumLandmarks(m_Widget->m_SpinBoxMaxLandmarks->value());
	icp->SetMaxIter(m_Widget->m_SpinBoxMaxIterations->value());
	icp->SetClosestPointFinder(cpf);

	int metric;
	switch (m_Widget->m_ComboBoxMetric->currentIndex())
	{
	case 0: metric = LOG_ABSOLUTE_DISTANCE; break;
	case 1: metric = ABSOLUTE_DISTANCE; break;
	case 2: metric = SQUARED_DISTANCE; break;
	}
	icp->SetMetric(metric);
	icp->Modified();
	icp->Update();

	// get the resulting transformation matrix (this matrix takes the source
	// points to the target points)
	vtkSmartPointer<vtkMatrix4x4> m = icp->GetMatrix();
	m_PreviousTransformMatrix->DeepCopy(m);
	//std::cout << "ICP transform: " << *m << std::endl;

	m_Widget->m_lcdNumberICPIterations->display(icp->GetNumIter());
	m_Widget->m_lcdNumberICPError->display(icp->GetMeanDist());

	// do the transform
	vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter =
		vtkSmartPointer<vtkTransformPolyDataFilter>::New();

	icpTransformFilter->SetInput(m_Data);
	icpTransformFilter->SetTransform(icp);
	icpTransformFilter->Update();

	// update m_PreviousFrame
	m_PreviousFrame->ShallowCopy(icpTransformFilter->GetOutput());

	// append world and new data
	vtkSmartPointer<vtkAppendPolyData> appendFilter =
		vtkSmartPointer<vtkAppendPolyData>::New();
	appendFilter->AddInput(m_TheWorld);
	appendFilter->AddInput(m_PreviousFrame);
	appendFilter->Update();
	m_TheWorld->ShallowCopy(appendFilter->GetOutput());

	// get rid of useless cells
	vtkSmartPointer<vtkCellArray> cells =
		vtkSmartPointer<vtkCellArray>::New();
	for (vtkIdType i = 0; i < m_TheWorld->GetNumberOfPoints(); i++)
	{
		cells->InsertNextCell(1, &i);
	}
	m_TheWorld->SetVerts(cells);
	m_TheWorld->Update();

	// update number of points label
	m_Widget->m_lcdNumberPointsInWorld->display(m_TheWorld->GetNumberOfPoints());

	if (update)
	{
		// update scene
		m_DataActor3D->SetData(m_TheWorld, false);

		emit UpdateGUI();
	}

	// cleanup
	delete cpf;
}
//----------------------------------------------------------------------------
void
StitchingPlugin::SaveVTKData()
{
	QString outputFile = QFileDialog::getSaveFileName(m_Widget, "Save VTK File", "D:/RITK/bin/release/Data/", "VTK files (*.vtk)");		

	if (!outputFile.isEmpty()) 
	{
		vtkSmartPointer<vtkPolyDataWriter> writer =
			vtkSmartPointer<vtkPolyDataWriter>::New();
		writer->SetFileName(outputFile.toStdString().c_str());
		writer->SetInput(m_DataActor3D->GetData());
		writer->SetFileTypeToASCII();
		writer->Update();		
	}	
}