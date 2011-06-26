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

// our includes
#include <ExtendedICPTransform.h>
#include <ClosestPointFinder.h>
#include <ClosestPointFinderBruteForceCPU.h>
#include <ClosestPointFinderBruteForceGPU.h>
#include <ClosestPointFinderRBCCPU.h>
#include <ClosestPointFinderRBCGPU.h>
#include <ClosestPointFinderRBCCayton.h>
#include <defs.h>

StitchingPlugin::StitchingPlugin()
{
	// create the widget
	m_Widget = new StitchingWidget();

	// basic signals
	connect(this, SIGNAL(UpdateGUI()), m_Widget,							SLOT(UpdateGUI()));
	connect(this, SIGNAL(UpdateGUI()), m_Widget->m_VisualizationWidget3D,	SLOT(UpdateGUI()));

	// our signals and slots
	connect(m_Widget->m_PushButtonStitchFrame,				SIGNAL(clicked()),								this, SLOT(LoadStitch()));
	connect(m_Widget->m_PushButtonDelaunay2D,				SIGNAL(clicked()),								this, SLOT(Delaunay2DSelectedActors()));
	connect(m_Widget->m_PushButtonSaveVTKData,				SIGNAL(clicked()),								this, SLOT(SaveSelectedActors()));
	connect(m_Widget->m_HorizontalSliderPointSize,			SIGNAL(valueChanged(int)),						this, SLOT(ChangePointSize()));
	connect(m_Widget->m_ToolButtonChooseBackgroundColor1,	SIGNAL(clicked()),								this, SLOT(ChangeBackgroundColor1()));
	connect(m_Widget->m_ToolButtonChooseBackgroundColor2,	SIGNAL(clicked()),								this, SLOT(ChangeBackgroundColor2()));
	connect(m_Widget->m_ListWidgetHistory,					SIGNAL(itemSelectionChanged()),					this, SLOT(ShowHideActors()));
	connect(m_Widget->m_CheckBoxShowSelectedActors,			SIGNAL(stateChanged(int)),						this, SLOT(ShowHideActors()));
	connect(m_Widget->m_PushButtonHistoryDelete,			SIGNAL(clicked()),								this, SLOT(DeleteSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryMerge,				SIGNAL(clicked()),								this, SLOT(MergeSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryClean,				SIGNAL(clicked()),								this, SLOT(CleanSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryStitchSelection,	SIGNAL(clicked()),								this, SLOT(StitchSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryUndoTransform,		SIGNAL(clicked()),								this, SLOT(UndoTransformForSelectedActors()));
	connect(m_Widget->m_ListWidgetHistory,					SIGNAL(itemDoubleClicked(QListWidgetItem*)),	this, SLOT(HighlightActor(QListWidgetItem*)));

	// progress bar signals
	connect(this, SIGNAL(UpdateProgressBar(int)),		m_Widget->m_ProgressBar, SLOT(setValue(int)));
	connect(this, SIGNAL(InitProgressBar(int, int)),	m_Widget->m_ProgressBar, SLOT(setRange(int, int)));

	// signal for time measurements
	connect(this, SIGNAL(UpdateStats()), this, SLOT(ComputeStats()));

	// add data actor
	m_DataActor3D = vtkSmartPointer<ritk::RImageActorPipeline>::New();	
	m_DataActor3D->SetVisualizationMode(ritk::RImageActorPipeline::RGB);

	// initialize member objects
	m_Data = vtkSmartPointer<vtkPolyData>::New();
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
StitchingPlugin::ProcessEvent(ritk::Event::Pointer EventP)
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
			if (m_Widget->m_RadioButtonRecord->isChecked() && m_FramesProcessed % m_Widget->m_SpinBoxFrameStep->value() == 0)
			{
				try
				{
					LoadFrame();
				} catch (std::exception &e)
				{
					std::cout << "Exception: \"" << e.what() << "\""<< std::endl;
					return;
				}
				
				// add next actor
				int listSize = m_Widget->m_ListWidgetHistory->count();
				HistoryListItem* hli = new HistoryListItem();
				hli->setText(QDateTime::currentDateTime().time().toString("hh:mm:ss:zzz"));
				hli->m_actor = vtkSmartPointer<ritk::RImageActorPipeline>::New();
				hli->m_actor->SetData(m_Data, true);
				hli->m_transform = vtkSmartPointer<vtkMatrix4x4>::New();
				hli->m_transform->Identity();
				m_Widget->m_ListWidgetHistory->insertItem(listSize, hli);
				//hli->setSelected(true);

				m_Widget->m_CheckBoxShowSelectedActors->setText(QString("Show ") + QString::number(m_Widget->m_ListWidgetHistory->selectedItems().count()) + 
					QString("/") + QString::number(m_Widget->m_ListWidgetHistory->count()) + " Actors");
			}
			else if (m_Widget->m_RadioButtonLiveStitching->isChecked() && m_FramesProcessed % m_Widget->m_SpinBoxFrameStep->value() == 0)
			{
				try
				{
					LoadStitch();
				} catch (std::exception &e)
				{
					std::cout << "Exception: \"" << e.what() << "\""<< std::endl;
					return;
				}

				m_Widget->m_CheckBoxShowSelectedActors->setText(QString("Show ") + QString::number(m_Widget->m_ListWidgetHistory->selectedItems().count()) + 
					QString("/") + QString::number(m_Widget->m_ListWidgetHistory->count()) + " Actors");
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
StitchingPlugin::ShowHideActors()
{
	int numPoints = 0;
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected() && m_Widget->m_CheckBoxShowSelectedActors->isChecked())
		{
			m_Widget->m_VisualizationWidget3D->AddActor(hli->m_actor);
		} else
		{
			m_Widget->m_VisualizationWidget3D->RemoveActor(hli->m_actor);
		}

		if (hli->isSelected())
		{
			numPoints += hli->m_actor->GetData()->GetNumberOfPoints();
		}
	}

	// show number of points for selected history entries
	m_Widget->m_LabelNumberOfPoints->setText(QString::number(numPoints));

	m_Widget->m_CheckBoxShowSelectedActors->setText(QString("Show ") + QString::number(m_Widget->m_ListWidgetHistory->selectedItems().count()) + 
		QString("/") + QString::number(m_Widget->m_ListWidgetHistory->count()) + " Actors");
}
void
StitchingPlugin::HighlightActor(QListWidgetItem* item)
{
	HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(item);

	// toggle colormode
	int mode = hli->m_actor->GetMapper()->GetColorMode();

	if (mode == 0)
	{
		hli->m_actor->GetMapper()->ColorByArrayComponent(0, 3);
		hli->m_actor->GetMapper()->SetColorModeToMapScalars();
		hli->setTextColor(QColor(0, 50, 255, 255));
	} else
	{
		hli->m_actor->GetMapper()->SetColorModeToDefault();
		hli->setTextColor(QColor(0, 0, 0, 255));
	}
}
void
StitchingPlugin::DeleteSelectedActors()
{
	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);

	// save all indices of list for actors that have to be deleted
	std::vector<int> toDelete;

	// delete all selected steps from history and from memory
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			DBG << "deleting actor " << i << std::endl;

			m_Widget->m_VisualizationWidget3D->RemoveActor(hli->m_actor);
			toDelete.push_back(i);	

			emit UpdateProgressBar(counterProgressBar++);
			QCoreApplication::processEvents();
		}
	}

	// run backwards through toDelete and delete those items from History and from memory
	for (int i = toDelete.size() - 1; i >= 0; --i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->takeItem(toDelete.at(i)));
		delete hli;
	}

	// update gui
	int size = m_Widget->m_ListWidgetHistory->count();
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

			DBG << "merge - adding actor " << i << std::endl;
		}
	}

	// nothing selected...
	if (firstIndex == -1)
	{
		delete hli;
		return;
	}

	try
	{
		appendFilter->Update();
		hli->m_actor->SetData(appendFilter->GetOutput(), true);
	} catch (std::exception &e)
	{
		QMessageBox msgBox(QMessageBox::Critical, QString("Error"), 
			QString("Try 'Clean' or selecting a smaller number of actors to merge!\n\"") + QString(e.what()) + QString("\""), 
			QMessageBox::Ok);
		msgBox.exec();

		return;
	}

	// get rid of useless cells
	vtkSmartPointer<vtkCellArray> cells =
		vtkSmartPointer<vtkCellArray>::New();
	for (vtkIdType i = 0; i < hli->m_actor->GetData()->GetNumberOfPoints(); i++)
	{
		cells->InsertNextCell(1, &i);
	}
	hli->m_actor->GetData()->SetVerts(cells);
	hli->m_actor->GetData()->Update();

	// and add the new merged history entry at a reasonable position
	m_Widget->m_ListWidgetHistory->insertItem(firstIndex, hli);

	// clean the selected history entries
	DeleteSelectedActors();

	// set the new element as selected
	hli->setSelected(true);
}
void
StitchingPlugin::CleanSelectedActors()
{
	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);

	int numPoints = 0;
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			DBG << "cleaning actor " << i << std::endl;

			try
			{
				Clean(hli->m_actor->GetData());
			} catch (std::exception &e)
			{
				std::cout << "Exception: \"" << e.what() << "\""<< std::endl;
				continue;
			}

			numPoints += hli->m_actor->GetData()->GetNumberOfPoints();

			emit UpdateProgressBar(++counterProgressBar);
			QCoreApplication::processEvents();
		}
	}

	// show number of points for selected history entries
	m_Widget->m_LabelNumberOfPoints->setText(QString::number(numPoints));

	emit UpdateGUI();
}
void
StitchingPlugin::StitchSelectedActors()
{
	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);

	QTime t;
	int overallTime = 0;

	// stitch all selected actors to previous actor in list
	// ignore the first one, because it can not be stitched to anything
	for (int i = 1; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			DBG << "stitching actor " << i << std::endl;
			if (m_Widget->m_CheckBoxShowSelectedActors->isChecked())
			{
				HighlightActor(hli);
			}
			//emit UpdateGUI();
			QCoreApplication::processEvents();

			HistoryListItem* hli_prev = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i - 1));

			t.start();
			try
			{
				QTime timeOverall;
				timeOverall.start();
				Stitch(hli->m_actor->GetData(), hli_prev->m_actor->GetData(), hli_prev->m_transform, hli->m_actor->GetData(), hli->m_transform);
				OVERALL_TIME = timeOverall.elapsed();
				LOAD_TIME = 0; // because frames are already loaded
			} catch (std::exception &e)
			{
				std::cout << "Exception: \"" << e.what() << "\"" << std::endl;
				continue;
			}
			overallTime += t.elapsed();

			if (m_Widget->m_CheckBoxShowSelectedActors->isChecked())
			{
				HighlightActor(hli);
			}
			emit UpdateStats();
			emit UpdateProgressBar(++counterProgressBar);
			emit UpdateGUI();
			QCoreApplication::processEvents();
		}
	}

	std::cout << overallTime << " ms for Stitching " << m_Widget->m_ListWidgetHistory->selectedItems().count() << " frames (time to update GUI excluded)" << std::endl;

	emit UpdateGUI();
}
void
StitchingPlugin::UndoTransformForSelectedActors()
{
	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);

	// take the inverse of the transform s.t. we get the original data back (approximately)
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			DBG << "undoing transform for actor " << i << std::endl;

			// invert the transform
			hli->m_transform->Invert();

			// set up transform
			vtkSmartPointer<vtkTransform> inverseTransform =
				vtkSmartPointer<vtkTransform>::New();

			inverseTransform->SetMatrix(hli->m_transform);
			inverseTransform->Modified();

			// set up transform filter
			vtkSmartPointer<vtkTransformPolyDataFilter> inverseTransformFilter =
				vtkSmartPointer<vtkTransformPolyDataFilter>::New();

			inverseTransformFilter->SetInput(hli->m_actor->GetData());
			inverseTransformFilter->SetTransform(inverseTransform);
			inverseTransformFilter->Modified();
			inverseTransformFilter->Update();

			// update the data
			hli->m_actor->GetData()->DeepCopy(inverseTransformFilter->GetOutput());

			// set transform to identity
			hli->m_transform->Identity();

			emit UpdateProgressBar(++counterProgressBar);
			emit UpdateGUI();
			QCoreApplication::processEvents();
		}
	}

	emit UpdateGUI();
}
void
StitchingPlugin::SaveSelectedActors()
{
	// get output file
	QString outputFile = QFileDialog::getSaveFileName(m_Widget, QString("Save ") + 
		QString::number(m_Widget->m_ListWidgetHistory->selectedItems().size()) + 
		" selected actors in seperate VTK files", "D:/RITK/bin/release/Data/", "VTK files (*.vtk)");	
	if (outputFile.isEmpty())
		return;

	// get rid of ".vtk"
	outputFile.remove(outputFile.size() - 4, 4);


	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);


	int fileCount = 0;
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			QString partFileName = outputFile + QString::number(fileCount++) + ".vtk";
			DBG << "saving actor " << i << " to " << partFileName.toStdString() << std::endl;
			vtkSmartPointer<vtkPolyDataWriter> writer =
				vtkSmartPointer<vtkPolyDataWriter>::New();
			writer->SetFileName(partFileName.toStdString().c_str());
			writer->SetInput(hli->m_actor->GetData());
			writer->SetFileTypeToASCII();
			writer->Update();

			emit UpdateProgressBar(++counterProgressBar);
			QCoreApplication::processEvents();
		} 
	}			
}
void
StitchingPlugin::Delaunay2DSelectedActors()
{
	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);

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

			try
			{
				Delaunay2D->Update();
				hli->m_actor->SetData(Delaunay2D->GetOutput(), true);
			} catch (std::exception &e)
			{
				std::cout << "Exception: \"" << e.what() << "\""<< std::endl;
				continue;
			}

			emit UpdateProgressBar(++counterProgressBar);
			QCoreApplication::processEvents();
		} 
	}

	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::ComputeStats()
{
	std::stringstream visualPercentage;

	int percentageLoad = static_cast<int>((static_cast<double>(LOAD_TIME) / static_cast<double>(OVERALL_TIME)) * 100.0);
	for (int i = 0; i < percentageLoad / 3; ++i)
		visualPercentage << "|";
	std::string visualPercentageLoad = visualPercentage.str();
	visualPercentage.str(""); visualPercentage.clear();

	int percentageClip = static_cast<int>((static_cast<double>(CLIP_TIME) / static_cast<double>(OVERALL_TIME)) * 100.0);
	for (int i = 0; i < percentageClip / 3; ++i)
		visualPercentage << "|";
	std::string visualPercentageClip = visualPercentage.str();
	visualPercentage.str(""); visualPercentage.clear();

	int percentageICP = static_cast<int>((static_cast<double>(ICP_TIME) / static_cast<double>(OVERALL_TIME)) * 100.0);
	for (int i = 0; i < percentageICP / 3; ++i)
		visualPercentage << "|";
	std::string visualPercentageICP = visualPercentage.str();
	visualPercentage.str(""); visualPercentage.clear();

	int percentageTransform = static_cast<int>((static_cast<double>(TRANSFORM_TIME) / static_cast<double>(OVERALL_TIME)) * 100.0);
	for (int i = 0; i < percentageTransform / 3; ++i)
		visualPercentage << "|";
	std::string visualPercentageTransform = visualPercentage.str();
	visualPercentage.str(""); visualPercentage.clear();

	m_Widget->m_LabelTimeLoad->setText(QString(QString(visualPercentageLoad.c_str()) + QString(" (") + QString::number(LOAD_TIME) + QString(" ms, ") + QString::number(percentageLoad) + QString("%)")));
	m_Widget->m_LabelTimeClip->setText(QString(QString(visualPercentageClip.c_str()) + QString(" (") + QString::number(CLIP_TIME) + QString(" ms, ") + QString::number(percentageClip) + QString("%)")));
	m_Widget->m_LabelTimeICP->setText(QString(QString(visualPercentageICP.c_str()) + QString(" (") + QString::number(ICP_TIME) + QString(" ms, ") + QString::number(percentageICP) + QString("%)")));
	m_Widget->m_LabelTimeTransform->setText(QString(QString(visualPercentageTransform.c_str()) + QString(" (") + QString::number(TRANSFORM_TIME) + QString(" ms, ") + QString::number(percentageTransform) + QString("%)")));
	m_Widget->m_LabelTimeOverall->setText(QString(QString::number(OVERALL_TIME) + QString(" ms")));
}
void
StitchingPlugin::LoadInitialize()
{
	LoadFrame();
	InitializeHistory();
}
void
StitchingPlugin::LoadStitch()
{
	QTime timeOverall;
	timeOverall.start();

	// get last entry in history to determine previousTransformationMatrix and previousFrame
	int listSize = m_Widget->m_ListWidgetHistory->count();

	if (listSize == 0)
	{
		LoadInitialize();
		return;
	}

	// load the new frame
	QTime loadFrameTime;
	loadFrameTime.start();
	LoadFrame();
	LOAD_TIME = loadFrameTime.elapsed();

	// create new history entry
	HistoryListItem* hli = new HistoryListItem();
	hli->setText(QDateTime::currentDateTime().time().toString("hh:mm:ss:zzz"));
	hli->m_actor = vtkSmartPointer<ritk::RImageActorPipeline>::New();
	hli->m_actor->GetProperty()->SetPointSize(m_Widget->m_HorizontalSliderPointSize->value());
	hli->m_transform = vtkSmartPointer<vtkMatrix4x4>::New();

	// get the previous history entry
	HistoryListItem* hli_last = reinterpret_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(listSize - 1));

	// stitch to just loaded frame to the previous frame (given by last history entry)
	Stitch(m_Data, hli_last->m_actor->GetData(), hli_last->m_transform, hli->m_actor->GetData(), hli->m_transform);

	OVERALL_TIME = timeOverall.elapsed();

	m_Widget->m_ListWidgetHistory->insertItem(listSize, hli);
	hli->setSelected(true);

	if (m_Widget->m_CheckBoxShowSelectedActors->isChecked())
	{
		ShowHideActors();
	}

	// stats...
	emit UpdateStats();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::LoadFrame()
{
	m_DataActor3D->SetData(m_CurrentFrame);
	m_Data->DeepCopy(m_DataActor3D->GetData());

	// remove invalid points
	ExtractValidPoints();
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
		if (rand() % 4 != 0)
			continue;

		m_Data->GetPoint(i, p);

		if (p[0] == p[0]) // i.e. not QNAN
		{
			points->InsertNextPoint(p[0], p[1], p[2]);
			double* tmp = m_Data->GetPointData()->GetScalars()->GetTuple(i);
			colors->InsertNextTuple(tmp);
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
	double percentage = m_Widget->m_DoubleSpinBoxClipPercentage->value();

	if (percentage == 0.0)
		return;

	vtkSmartPointer<vtkClipPolyData> clipper =
		vtkSmartPointer<vtkClipPolyData>::New();	
	vtkSmartPointer<vtkBox> box =
		vtkSmartPointer<vtkBox>::New();

	double bounds[6];

	toBeClipped->GetBounds(bounds);

	// modify x, y (and z) bounds
	bounds[0] += percentage*(bounds[1] - bounds[0]);
	bounds[1] -= percentage*(bounds[1] - bounds[0]);
	bounds[2] += percentage*(bounds[3] - bounds[2]);
	bounds[3] -= percentage*(bounds[3] - bounds[2]);
	bounds[4] += percentage*(bounds[5] - bounds[4]);
	bounds[5] -= percentage*(bounds[5] - bounds[4]);

	box->SetBounds(bounds);
	clipper->SetClipFunction(box);
	clipper->InsideOutOn();
	clipper->SetInput(toBeClipped);
	clipper->Update();

	toBeClipped->ShallowCopy(clipper->GetOutput());
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
StitchingPlugin::CleanFrame()
{
	Clean(m_Data);
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
	cleanPolyData->SetInput(toBeCleaned);
	cleanPolyData->Update();

	toBeCleaned->DeepCopy(cleanPolyData->GetOutput());
}
//----------------------------------------------------------------------------
void
StitchingPlugin::InitializeHistory()
{
	// clear history
	m_Widget->m_ListWidgetHistory->selectAll();
	DeleteSelectedActors();

	// add first actor
	HistoryListItem* hli = new HistoryListItem();
	hli->setText(QDateTime::currentDateTime().time().toString("hh:mm:ss:zzz"));
	hli->m_actor = vtkSmartPointer<ritk::RImageActorPipeline>::New();
	hli->m_actor->SetData(m_Data, true);
	hli->m_transform = vtkSmartPointer<vtkMatrix4x4>::New();
	hli->m_transform->Identity();
	m_Widget->m_ListWidgetHistory->insertItem(0, hli);
	hli->setSelected(true);

	emit UpdateGUI();
}
//----------------------------------------------------------------------------
void
StitchingPlugin::Stitch(vtkPolyData* toBeStitched, vtkPolyData* previousFrame,
						vtkMatrix4x4* previousTransformationMatrix,
						vtkPolyData* outputStitchedPolyData,
						vtkMatrix4x4* outputTransformationMatrix)
{
	// get a subvolume of the original data
	vtkSmartPointer<vtkPolyData> voi = 
		vtkSmartPointer<vtkPolyData>::New();

	voi->ShallowCopy(toBeStitched);
	
	if (voi->GetNumberOfPoints() < m_Widget->m_SpinBoxLandmarks->value())
	{
		std::cout << "Reduce the number of landmarks!!! Data might be inconsistent now..." << std::endl;
		return;
	}

	// initialize ClosestPointFinder
	ClosestPointFinder* cpf; 
	switch (m_Widget->m_ComboBoxClosestPointFinder->currentIndex())
	{
	case 0: cpf = new ClosestPointFinderBruteForceGPU(m_Widget->m_SpinBoxLandmarks->value()); break;
	case 1: cpf = new ClosestPointFinderBruteForceCPU(m_Widget->m_SpinBoxLandmarks->value(), false); break;
	case 2: cpf = new ClosestPointFinderBruteForceCPU(m_Widget->m_SpinBoxLandmarks->value(), true); break;
	case 3: cpf = new ClosestPointFinderRBCCPU(m_Widget->m_SpinBoxLandmarks->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value())); break;
	case 4: cpf = new ClosestPointFinderRBCGPU(m_Widget->m_SpinBoxLandmarks->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value())); break;
	case 5: cpf = new ClosestPointFinderRBCCayton(m_Widget->m_SpinBoxLandmarks->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value())); break;
	}

	cpf->SetWeightRGB(static_cast<float>(m_Widget->m_DoubleSpinBoxRGBWeight->value()));

	int metric;
	switch (m_Widget->m_ComboBoxMetric->currentIndex())
	{
	case 0: metric = LOG_ABSOLUTE_DISTANCE; break;
	case 1: metric = ABSOLUTE_DISTANCE; break;
	case 2: metric = SQUARED_DISTANCE; break;
	}
	cpf->SetMetric(metric);

	// iterative closest point (ICP) transformation
	vtkSmartPointer<ExtendedICPTransform> icp = 
		vtkSmartPointer<ExtendedICPTransform>::New();

	// configure icp
	icp->SetSource(voi);
	icp->SetTarget(previousFrame);
	icp->SetNumLandmarks(m_Widget->m_SpinBoxLandmarks->value());
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaxMeanDist(static_cast<float>(m_Widget->m_DoubleSpinBoxMaxRMS->value()));
	icp->SetMaxIter(m_Widget->m_SpinBoxMaxIterations->value());
	icp->SetRemoveOutliers(m_Widget->m_CheckBoxRemoveOutliers->isChecked());
	icp->SetOutlierRate(static_cast<float>(m_Widget->m_DoubleSpinBoxOutlierRate->value()));
	icp->SetApplyPreviousTransform(m_Widget->m_CheckBoxUsePreviousTransformation->isChecked());
	icp->SetPreviousTransformMatrix(previousTransformationMatrix);

	// new stuff... strange
	double* bounds = voi->GetBounds();
	double boundDiagonal = sqrt((bounds[1] - bounds[0])*(bounds[1] - bounds[0]) + (bounds[3] - bounds[2])*(bounds[3] - bounds[2]) + (bounds[5] - bounds[4])*(bounds[5] - bounds[4]));
	icp->SetNormalizeRGBToDistanceValuesFactor(static_cast<float>(boundDiagonal / sqrt(3.0)));

	// transform vtkPolyData in our own structures and clip simultaneously
	QTime timeClip;
	timeClip.start();
	icp->vtkPolyDataToPointCoordsAndColors(m_Widget->m_DoubleSpinBoxClipPercentage->value());
	CLIP_TIME = timeClip.elapsed();

	QTime timeICP;
	timeICP.start();

	icp->SetClosestPointFinder(cpf);
	icp->Modified();
	icp->Update();

	ICP_TIME = timeICP.elapsed();

	// update icp runtime, etc
	m_Widget->m_LabelICPMeanTargetDistance->setText(QString::number(icp->GetMeanTargetDistance(), 'f', 2));
	

	QTime timeTransform;
	timeTransform.start();
	// update output parameter
	outputTransformationMatrix->DeepCopy(icp->GetMatrix());

	// perform the transform
	vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter =
		vtkSmartPointer<vtkTransformPolyDataFilter>::New();

	icpTransformFilter->SetInput(toBeStitched);
	icpTransformFilter->SetTransform(icp);
	icpTransformFilter->Update();

	outputStitchedPolyData->DeepCopy(icpTransformFilter->GetOutput());

	TRANSFORM_TIME = timeTransform.elapsed();

	// include the previous transformation into the matrix to allow for "undo"
	if (m_Widget->m_CheckBoxUsePreviousTransformation->isChecked())
	{
		//vtkMatrix4x4::Multiply4x4(outputTransformationMatrix, previousTransformationMatrix, outputTransformationMatrix);
	}

	// update debug information in GUI
	m_Widget->m_LabelICPIterations->setText(QString::number(icp->GetNumIter()));
	m_Widget->m_LabelICPError->setText(QString::number(icp->GetMeanDist()));

	// cleanup
	delete cpf; // delete ClosestPointFinder
}