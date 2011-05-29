// standard includes
#include "StitchingPlugin.h"
#include "DebugManager.h"
#include "Manager.h"

// cpp std
#include <vector>

// Qt includes
#include <QFileDialog>
#include <QTime>
#include <QColorDialog>

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
#include <defs.h>



StitchingPlugin::StitchingPlugin()
{
	// create the widget
	m_Widget = new StitchingWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget,							SLOT(UpdateGUI()));
	connect(this, SIGNAL(UpdateGUI()), m_Widget->m_VisualizationWidget3D,	SLOT(UpdateGUI()));

	// set signals (from buttons, etc) and slots (in this file)
	connect(m_Widget->m_PushButtonLoadCleanStitch,			SIGNAL(clicked()),								this, SLOT(LoadCleanStitch()));
	connect(m_Widget->m_PushButtonInitialize,				SIGNAL(clicked()),								this, SLOT(LoadCleanInitialize()));
	connect(m_Widget->m_PushButtonDelaunay2D,				SIGNAL(clicked()),								this, SLOT(Delaunay2D()));
	connect(m_Widget->m_PushButtonSaveVTKData,				SIGNAL(clicked()),								this, SLOT(SaveVTKData()));
	connect(m_Widget->m_HorizontalSliderPointSize,			SIGNAL(valueChanged(int)),						this, SLOT(ChangePointSize()));
	connect(m_Widget->m_ToolButtonChooseBackgroundColor1,	SIGNAL(clicked()),								this, SLOT(ChangeBackgroundColor1()));
	connect(m_Widget->m_ToolButtonChooseBackgroundColor2,	SIGNAL(clicked()),								this, SLOT(ChangeBackgroundColor2()));
	connect(m_Widget->m_ListWidgetHistory,					SIGNAL(itemSelectionChanged()),					this, SLOT(ShowHideActors()));
	connect(m_Widget->m_PushButtonHistoryDelete,			SIGNAL(clicked()),								this, SLOT(DeleteSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryMerge,				SIGNAL(clicked()),								this, SLOT(MergeSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryClean,				SIGNAL(clicked()),								this, SLOT(CleanSelectedActors()));
	connect(m_Widget->m_ListWidgetHistory,					SIGNAL(itemDoubleClicked(QListWidgetItem*)),	this, SLOT(HighlightActor(QListWidgetItem*)));
	
		
	// add data actor
	m_DataActor3D = vtkSmartPointer<ritk::RImageActorPipeline>::New();	
	m_DataActor3D->SetVisualizationMode(ritk::RImageActorPipeline::RGB);
	//m_Widget->m_VisualizationWidget3D->AddActor(m_DataActor3D);

	// initialize member objects
	m_Data =					vtkSmartPointer<vtkPolyData>::New();
	m_PreviousFrame =			vtkSmartPointer<vtkPolyData>::New();
	m_PreviousTransformMatrix =	vtkSmartPointer<vtkMatrix4x4>::New();
	m_FramesProcessed = 0;
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
StitchingPlugin::ChangePointSize()
{
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			hli->m_actor->GetProperty()->SetPointSize(m_Widget->m_HorizontalSliderPointSize->value());
		}
	}
	
	emit UpdateGUI();
}
void
StitchingPlugin::ChangeBackgroundColor1()
{
	QColor color = QColorDialog::getColor();

	m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->
		SetBackground(color.red()/255., color.green()/255., color.blue()/255.);
}
void
StitchingPlugin::ChangeBackgroundColor2()
{
	QColor color = QColorDialog::getColor();

	m_Widget->m_VisualizationWidget3D->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->
		SetBackground2(color.red()/255., color.green()/255., color.blue()/255.);
}
//----------------------------------------------------------------------------
void
StitchingPlugin::ShowHideActors()
{
	int numPoints = 0;
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			m_Widget->m_VisualizationWidget3D->AddActor(hli->m_actor);
			numPoints += hli->m_actor->GetData()->GetNumberOfPoints();
		} else
		{
			m_Widget->m_VisualizationWidget3D->RemoveActor(hli->m_actor);
		}
	}

	// show number of points for selected history entries
	m_Widget->m_LabelNumberOfPoints->setText(QString::number(numPoints));
}
void
StitchingPlugin::HighlightActor(QListWidgetItem* item)
{
	HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(item);

	// toggle colormode
	int mode = hli->m_actor->GetMapper()->GetColorMode();

	if (mode == 0)
	{
		hli->m_actor->GetMapper()->ColorByArrayComponent(0, 1);
		hli->m_actor->GetMapper()->SetColorModeToMapScalars();
		//hli->setBackgroundColor(QColor(0, 0, 255, 100));
		hli->setTextColor(QColor(0, 50, 255, 255));
	} else
	{
		hli->m_actor->GetMapper()->SetColorModeToDefault();
		//hli->setBackgroundColor(QColor(255, 255, 255, 255));
		hli->setTextColor(QColor(0, 0, 0, 255));
	}
	emit UpdateGUI();
}
void
StitchingPlugin::DeleteSelectedActors()
{
	// save all indices of list for actors that have to be deleted
	std::vector<int> toDelete;

	// delete all selected steps from history and from memory
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			m_Widget->m_VisualizationWidget3D->RemoveActor(hli->m_actor);
			toDelete.push_back(i);
		}
	}

	// run backwards through toDelete and delete those items from History and from memory
	for (int i = toDelete.size() - 1; i >= 0; --i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->takeItem(toDelete.at(i)));
		delete hli;
	}

	// update previousTransform and previousFrame to be the last still existing frame in the history list
	int size = m_Widget->m_ListWidgetHistory->count();
	if (size > 0)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(size - 1));
		m_PreviousFrame->DeepCopy(hli->m_actor->GetData());
		m_PreviousTransformMatrix->DeepCopy(hli->m_transform);
	}
}
void
StitchingPlugin::MergeSelectedActors()
{
	if (m_Widget->m_ListWidgetHistory->selectedItems().size() <= 1)
	{
		std::cout << "nothing to be merged..." << std::endl;
		return;
	}

	// append the whole history
	vtkSmartPointer<vtkAppendPolyData> appendFilter =
		vtkSmartPointer<vtkAppendPolyData>::New();

	// create the merged history entry
	HistoryListItem* hli = new HistoryListItem;
	hli->setText(QDateTime::currentDateTime().time().toString("hh:mm:ss:zzz") + " (merged)");
	hli->m_transform = vtkSmartPointer<vtkMatrix4x4>::New();
	hli->m_actor = vtkSmartPointer<ritk::RImageActorPipeline>::New();
	hli->setToolTip(QString("merged"));

	// at this index, the merged points will be taken afterwards
	int firstIndex = -1;

	// add the data of each actor to the appendFilter and store the last transformation
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli2 = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));

		if (hli2->isSelected())
		{
			appendFilter->AddInput(hli2->m_actor->GetData());

			hli->m_transform->DeepCopy(hli2->m_transform);

			if (firstIndex == -1)
			{
				firstIndex = i;
			}
		}
	}

	// nothing selected...
	if (firstIndex == -1)
	{
		delete hli;
		return;
	}

	appendFilter->Update();
	hli->m_actor->SetData(appendFilter->GetOutput(), true);

	// get rid of useless cells
	vtkSmartPointer<vtkCellArray> cells =
		vtkSmartPointer<vtkCellArray>::New();
	for (vtkIdType i = 0; i < hli->m_actor->GetData()->GetNumberOfPoints(); i++)
	{
		cells->InsertNextCell(1, &i);
	}
	hli->m_actor->GetData()->SetVerts(cells);
	hli->m_actor->GetData()->Update();

	// clean the selected history entries
	DeleteSelectedActors();

	// and add the new merged history entry at a reasonable position
	m_Widget->m_ListWidgetHistory->insertItem(firstIndex, hli);
	hli->setSelected(true);
}
void
StitchingPlugin::CleanSelectedActors()
{
	int numPoints = 0;
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			Clean(hli->m_actor->GetData());
			numPoints += hli->m_actor->GetData()->GetNumberOfPoints();
		}
	}

	// show number of points for selected history entries
	m_Widget->m_LabelNumberOfPoints->setText(QString::number(numPoints));

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
		if (m_Widget->m_SpinBoxFrameStep->value() != 0 && ++m_FramesProcessed % m_Widget->m_SpinBoxFrameStep->value() == 0)
		{
			LoadCleanStitch();
		}

		// enable buttons (ProcessEvent has to be called at least once before
		// we can load the data into our plugin)
		m_Widget->m_PushButtonInitialize->setEnabled(true);
		
		emit UpdateGUI();
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
//----------------------------------------------------------------------------
void
StitchingPlugin::LoadCleanInitialize()
{
	QTime t = QTime::currentTime();

	t.start();
	LoadFrame(false);
	std::cout << "LoadFrame():     " << t.elapsed() << " ms" << std::endl;

	t.start();
	CleanFrame(false);
	std::cout << "CleanFrame():    " << t.elapsed() << " ms" << std::endl;

	t.start();
	InitializeHistory();
	std::cout << "Initialize():    " << t.elapsed() << " ms" << std::endl;

	emit UpdateGUI();
}
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
	Stitch(false);
	std::cout << "Stitch(): " << t.elapsed() << " ms" << std::endl;

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

	m_Widget->m_PushButtonLoadCleanStitch->setEnabled(true);
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
StitchingPlugin::Clean(vtkPolyData *toBeCleaned)
{
	// stop if tolerance is 0 (almost nothing will be cleaned, but it still takes
	// some milliseconds... not good...
	if (m_Widget->m_DoubleSpinBoxCleanTolerance->value() < DBL_EPSILON)
		return;

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
StitchingPlugin::InitializeHistory()
{
	// copy data
	m_PreviousFrame->DeepCopy(m_Data);
	m_PreviousTransformMatrix->Identity();

	// clear history
	m_Widget->m_ListWidgetHistory->selectAll();
	DeleteSelectedActors();

	// add first actor
	HistoryListItem* hle = new HistoryListItem();
	hle->setText(QDateTime::currentDateTime().time().toString("hh:mm:ss:zzz"));
	hle->m_actor = vtkSmartPointer<ritk::RImageActorPipeline>::New();
	hle->m_actor->SetData(m_PreviousFrame, true);
	hle->m_transform = m_PreviousTransformMatrix;
	m_Widget->m_ListWidgetHistory->insertItem(0, hle);
	hle->setSelected(true);

	// enable buttons
	//m_Widget->m_PushButtonStitch->setEnabled(true);
	m_Widget->m_PushButtonDelaunay2D->setEnabled(true);
	m_Widget->m_PushButtonSaveVTKData->setEnabled(true);
	m_Widget->m_PushButtonLoadCleanStitch->setEnabled(true);
	m_Widget->m_SpinBoxFrameStep->setEnabled(true);


	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::Stitch(bool update)
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

	if (voi->GetNumberOfPoints() < m_Widget->m_SpinBoxLandmarks->value())
	{
		std::cout << "Reduce the number of landmarks!!! Data might be inconsistent now..." << std::endl;
		return;
	}

	// initialize ClosestPointFinder
	ClosestPointFinder* cpf = new ClosestPointFinderBruteForceCPU(m_Widget->m_SpinBoxLandmarks->value());
	cpf->SetUseRGBData(m_Widget->m_CheckBoxUseRGBData->isChecked());
	cpf->SetWeightRGB(m_Widget->m_DoubleSpinBoxRGBWeight->value());

	// configure icp
	icp->SetSource(voi);
	icp->SetTarget(m_PreviousFrame);
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaxMeanDist(m_Widget->m_DoubleSpinBoxMaxRMS->value());
	icp->SetNumLandmarks(m_Widget->m_SpinBoxLandmarks->value());
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

	// update debug information in GUI
	m_Widget->m_LabelICPIterations->setText(QString::number(icp->GetNumIter()));
	m_Widget->m_LabelICPError->setText(QString::number(icp->GetMeanDist()));

	// do the transform
	vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter =
		vtkSmartPointer<vtkTransformPolyDataFilter>::New();

	icpTransformFilter->SetInput(m_Data);
	icpTransformFilter->SetTransform(icp);
	icpTransformFilter->Update();

	// update m_PreviousFrame
	m_PreviousFrame->ShallowCopy(icpTransformFilter->GetOutput());

	// create new history entry
	int listSize = m_Widget->m_ListWidgetHistory->count();
	HistoryListItem* hli = new HistoryListItem();
	hli->setText(QDateTime::currentDateTime().time().toString("hh:mm:ss:zzz"));
	hli->m_actor = vtkSmartPointer<ritk::RImageActorPipeline>::New();
	hli->m_actor->SetData(m_PreviousFrame, true);
	hli->m_transform = m_PreviousTransformMatrix;
	m_Widget->m_ListWidgetHistory->insertItem(listSize, hli);
	hli->setSelected(true);

	// cleanup
	delete cpf; // delete ClosestPointFinder
}
//----------------------------------------------------------------------------
void
StitchingPlugin::SaveVTKData()
{
	QString outputFile = QFileDialog::getSaveFileName(m_Widget, "Save VTK File", "D:/RITK/bin/release/Data/", "VTK files (*.vtk)");		

	// get last entry in history
	HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(m_Widget->m_ListWidgetHistory->count() - 1));

	if (!outputFile.isEmpty()) 
	{
		vtkSmartPointer<vtkPolyDataWriter> writer =
			vtkSmartPointer<vtkPolyDataWriter>::New();
		writer->SetFileName(outputFile.toStdString().c_str());
		writer->SetInput(hli->m_actor->GetData());
		writer->SetFileTypeToASCII();
		writer->Update();		
	}	
}
void
StitchingPlugin::Delaunay2D()
{
	// triangulation for each selected history entry - may take some time...
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			vtkSmartPointer<vtkTransform> t = 
				vtkSmartPointer<vtkTransform>::New();
			vtkSmartPointer<vtkDelaunay2D> Delaunay2D = 
				vtkSmartPointer<vtkDelaunay2D>::New();
			Delaunay2D->SetInput(hli->m_actor->GetData());
			Delaunay2D->SetTransform(t);
			Delaunay2D->Update();
			hli->m_actor->SetData(Delaunay2D->GetOutput(), true);
		} 
	}

	emit UpdateGUI();
}