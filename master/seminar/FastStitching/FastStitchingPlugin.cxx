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
#include <ClosestPointFinderBruteForceGPU.h>
#include <ClosestPointFinderRBCGPU.h>
#include <defs.h>

extern "C"
void
CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray, int w, int h);

FastStitchingPlugin::FastStitchingPlugin()
{
	// create the widget
	m_Widget = new FastStitchingWidget();

	// basic signals
	connect(this, SIGNAL(UpdateGUI()), m_Widget,							SLOT(UpdateGUI()));
	connect(this, SIGNAL(UpdateGUI()), m_Widget->m_VisualizationWidget3D,	SLOT(UpdateGUI()));

	// our signals and slots
	connect(m_Widget->m_PushButtonStitchFrame,				SIGNAL(clicked()),								this, SLOT(LoadStitch()));
	connect(m_Widget->m_HorizontalSliderPointSize,			SIGNAL(valueChanged(int)),						this, SLOT(ChangePointSize()));
	connect(m_Widget->m_ToolButtonChooseBackgroundColor1,	SIGNAL(clicked()),								this, SLOT(ChangeBackgroundColor1()));
	connect(m_Widget->m_ToolButtonChooseBackgroundColor2,	SIGNAL(clicked()),								this, SLOT(ChangeBackgroundColor2()));
	connect(m_Widget->m_ListWidgetHistory,					SIGNAL(itemSelectionChanged()),					this, SLOT(ShowHideActors()));
	connect(m_Widget->m_CheckBoxShowSelectedActors,			SIGNAL(stateChanged(int)),						this, SLOT(ShowHideActors()));
	connect(m_Widget->m_PushButtonHistoryStitchSelection,	SIGNAL(clicked()),								this, SLOT(StitchSelectedActors()));
	connect(m_Widget->m_PushButtonHistoryUndoTransform,		SIGNAL(clicked()),								this, SLOT(UndoTransformForSelectedActors()));
	connect(m_Widget->m_ComboBoxClosestPointFinder,			SIGNAL(currentIndexChanged(int)),				this, SLOT(ResetICPandCPF()));
	connect(m_Widget->m_SpinBoxLandmarks,					SIGNAL(valueChanged(int)),						this, SLOT(ResetICPandCPF()));
	connect(m_Widget->m_DoubleSpinBoxRGBWeight,				SIGNAL(valueChanged(double)),					this, SLOT(ResetICPandCPF()));
	connect(m_Widget->m_ComboBoxMetric,						SIGNAL(currentIndexChanged(int)),				this, SLOT(ResetICPandCPF()));
	connect(m_Widget->m_SpinBoxBufferSize,					SIGNAL(valueChanged(int)),						this, SLOT(ClearBuffer()));
	connect(this,											SIGNAL(RecordFrameAvailable()),					this, SLOT(RecordFrame()));
	connect(this,											SIGNAL(LiveFastStitchingFrameAvailable()),		this, SLOT(LiveFastStitching()));
	connect(m_Widget->m_DoubleSpinBoxHistDiffThresh,		SIGNAL(valueChanged(double)),					this, SLOT(SetThreshold(double)));


	// progress bar signals
	connect(this, SIGNAL(UpdateProgressBar(int)),		m_Widget->m_ProgressBar, SLOT(setValue(int)));
	connect(this, SIGNAL(InitProgressBar(int, int)),	m_Widget->m_ProgressBar, SLOT(setRange(int, int)));

	// signal for time measurements
	connect(this, SIGNAL(UpdateStats()), this, SLOT(ComputeStats()));

	// initialize member objects
	m_Data = vtkSmartPointer<vtkPolyData>::New();

	m_RangeTextureData = new unsigned char[FRAME_SIZE_X*FRAME_SIZE_Y];
	m_WCs = new float4[FRAME_SIZE_X*FRAME_SIZE_Y];

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cutilSafeCall(cudaMallocArray(&m_InputImgArr, &ChannelDesc, FRAME_SIZE_X, FRAME_SIZE_Y));
	cutilSafeCall(cudaMalloc((void**)&(m_devWCs), FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float4)));

	m_FramesProcessed = 0;

	// iterative closest point (ICP) transformation
	m_icp = vtkSmartPointer<ExtendedICPTransform>::New();

	// initialize ClosestPointFinder
	switch (m_Widget->m_ComboBoxClosestPointFinder->currentIndex())
	{
	case 0: m_cpf = new ClosestPointFinderRBCGPU(m_Widget->m_SpinBoxLandmarks->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value())); break;
	case 1: m_cpf = new ClosestPointFinderBruteForceGPU(m_Widget->m_SpinBoxLandmarks->value()); break;
	}

	// dirty hack - we just initialized m_icp and m_cpf, but the plugin crashes if we don't do it again (Qt contexts?!)
	m_ResetICPandCPFRequired = true;

	// for the bounded buffer
	m_BufferCounter = 0;
	m_BufferSize = m_Widget->m_SpinBoxBufferSize->value();

	// comparison of two consecutive frames
	m_CurrentHist = NULL;
	m_PreviousHist = NULL;
	m_HistogramDifferenceThreshold = m_Widget->m_DoubleSpinBoxHistDiffThresh->value();
}

FastStitchingPlugin::~FastStitchingPlugin()
{
	delete[] m_RangeTextureData;
	delete[] m_WCs;

	// Free GPU Memory that holds the previous and current world data
	cutilSafeCall(cudaFreeArray(m_InputImgArr));
	cutilSafeCall(cudaFree(m_devWCs));

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
FastStitchingPlugin::LiveFastStitching()
{
	m_Mutex.lock();
	try
	{
		LoadStitch();
	} catch (std::exception &e)
	{
		std::cout << "Exception: \"" << e.what() << "\""<< std::endl;
		m_Mutex.unlock();
		return;
	}

	m_Widget->m_CheckBoxShowSelectedActors->setText(QString("Show ") + QString::number(m_Widget->m_ListWidgetHistory->selectedItems().count()) + 
		QString("/") + QString::number(m_Widget->m_ListWidgetHistory->count()) + " Actors");

	m_Mutex.unlock();
}
void
FastStitchingPlugin::ClearBuffer()
{
	if (!m_Mutex.tryLock())
	{
		std::cout << "Stop FastStitching to clear the buffer" << std::endl;
		return;
	}	

	m_Widget->m_ListWidgetHistory->selectAll();
	DeleteSelectedActors();
	m_BufferCounter = 0;
	m_BufferSize = m_Widget->m_SpinBoxBufferSize->value();
	m_Mutex.unlock();
}

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
FastStitchingPlugin::ShowHideActors()
{
	int numPoints = 0;
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
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
FastStitchingPlugin::DeleteSelectedActors()
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
		HistoryListItem* hli = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			m_Widget->m_VisualizationWidget3D->RemoveActor(hli->m_actor);
			toDelete.push_back(i);	

			emit UpdateProgressBar(counterProgressBar++);
			QCoreApplication::processEvents();
		}
	}

	// run backwards through toDelete and delete those items from History and from memory
	for (int i = toDelete.size() - 1; i >= 0; --i)
	{
		HistoryListItem* hli = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->takeItem(toDelete.at(i)));
		delete hli;
	}

	// sync buffer
	int size = m_Widget->m_ListWidgetHistory->count();
	m_BufferCounter = size-1;
}
void
FastStitchingPlugin::StitchSelectedActors()
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
		HistoryListItem* hli = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
			HistoryListItem* hli_prev = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i - 1));

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

			emit UpdateStats();
			emit UpdateProgressBar(++counterProgressBar);
			emit UpdateGUI();
			QCoreApplication::processEvents();
		}
	}

	std::cout << overallTime << " ms for FastStitching " << m_Widget->m_ListWidgetHistory->selectedItems().count() << " frames (time to update GUI excluded)" << std::endl;

	emit UpdateGUI();
}
void
FastStitchingPlugin::UndoTransformForSelectedActors()
{
	int numSelected = m_Widget->m_ListWidgetHistory->selectedItems().count();
	int counterProgressBar = 1;
	emit InitProgressBar(0, numSelected);
	emit UpdateProgressBar(counterProgressBar);

	// take the inverse of the transform s.t. we get the original data back (approximately)
	for (int i = 0; i < m_Widget->m_ListWidgetHistory->count(); ++i)
	{
		HistoryListItem* hli = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(i));
		if (hli->isSelected())
		{
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
//----------------------------------------------------------------------------
void
FastStitchingPlugin::ComputeStats()
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
FastStitchingPlugin::LoadInitialize()
{
	LoadFrame();
	InitializeHistory();
}
void
FastStitchingPlugin::LoadStitch()
{
	QTime timeOverall;
	timeOverall.start();

	// get last entry in history to determine previousTransformationMatrix and previousFrame
	int listSize = m_Widget->m_ListWidgetHistory->count();

	if (listSize == 0)
	{
		LoadInitialize();
		m_BufferCounter = 0;
		return;
	}

	if (listSize < m_BufferSize)
		m_BufferCounter %= m_BufferSize;

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
	HistoryListItem* hli_last = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item(m_BufferCounter % m_BufferSize));

	if (m_BufferCounter >= m_BufferSize-1)
	{
		HistoryListItem* toBeDeleted = static_cast<HistoryListItem*>(m_Widget->m_ListWidgetHistory->item((m_BufferCounter+1) % m_BufferSize));
		m_Widget->m_VisualizationWidget3D->RemoveActor(toBeDeleted->m_actor);
		delete toBeDeleted;
	}

	// stitch to just loaded frame to the previous frame (given by last history entry)
	Stitch(m_Data, hli_last->m_actor->GetData(), hli_last->m_transform, hli->m_actor->GetData(), hli->m_transform);

	OVERALL_TIME = timeOverall.elapsed();

	m_Widget->m_ListWidgetHistory->insertItem((m_BufferCounter+1) % m_BufferSize, hli);
	hli->setSelected(true);

	if (m_Widget->m_CheckBoxShowSelectedActors->isChecked())
	{
		ShowHideActors();
	}

	// stats...
	emit UpdateStats();
	QCoreApplication::processEvents();

	++m_BufferCounter;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::LoadFrame()
{
	// Copy the input data to the device
	cutilSafeCall(cudaMemcpyToArray(m_InputImgArr, 0, 0, m_CurrentFrame->GetRangeImage()->GetBufferPointer(), FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float), cudaMemcpyHostToDevice));

	// Compute the world coordinates
	CUDARangeToWorld(m_devWCs, m_InputImgArr, FRAME_SIZE_X, FRAME_SIZE_Y);

	cutilSafeCall(cudaMemcpy(m_WCs, m_devWCs, FRAME_SIZE_X*FRAME_SIZE_Y*sizeof(float4), cudaMemcpyDeviceToHost));

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

	// update m_Data
	m_Data->SetPoints(points);
	m_Data->GetPointData()->SetScalars(colors);
	m_Data->SetVerts(cells);
	m_Data->Update();

	if (m_Data->GetNumberOfPoints() < m_Widget->m_SpinBoxLandmarks->value())
	{
		throw std::exception("not enough points for stitching!");
	}
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::InitializeHistory()
{
	// clear history
	m_Widget->m_ListWidgetHistory->selectAll();

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
FastStitchingPlugin::ResetICPandCPF() 
{
	m_ResetICPandCPFRequired = true;
}
//----------------------------------------------------------------------------
void
FastStitchingPlugin::Stitch(vtkPolyData* toBeStitched, vtkPolyData* previousFrame,
						vtkMatrix4x4* previousTransformationMatrix,
						vtkPolyData* outputStitchedPolyData,
						vtkMatrix4x4* outputTransformationMatrix)
{
	if (m_ResetICPandCPFRequired)
	{
		switch (m_Widget->m_ComboBoxClosestPointFinder->currentIndex())
		{
		case 0: m_cpf = new ClosestPointFinderRBCGPU(m_Widget->m_SpinBoxLandmarks->value(), static_cast<float>(m_Widget->m_DoubleSpinBoxNrOfRepsFactor->value())); break;
		case 1: m_cpf = new ClosestPointFinderBruteForceGPU(m_Widget->m_SpinBoxLandmarks->value()); break;
		}

		int metric;
		switch (m_Widget->m_ComboBoxMetric->currentIndex())
		{
		case 0: metric = LOG_ABSOLUTE_DISTANCE; break;
		case 1: metric = ABSOLUTE_DISTANCE; break;
		case 2: metric = SQUARED_DISTANCE; break;
		}

		m_cpf->SetMetric(metric);
		m_cpf->SetWeightRGB(m_Widget->m_DoubleSpinBoxRGBWeight->value());

		m_icp = vtkSmartPointer<ExtendedICPTransform>::New();
		m_icp->SetClosestPointFinder(m_cpf);
		m_icp->GetLandmarkTransform()->SetModeToRigidBody();
		m_icp->SetNumLandmarks(m_Widget->m_SpinBoxLandmarks->value());

		m_BufferSize = m_Widget->m_SpinBoxBufferSize->value();

		m_ResetICPandCPFRequired = false;
	}

	m_icp->SetSource(toBeStitched);
	m_icp->SetTarget(previousFrame);
	m_icp->SetMaxMeanDist(static_cast<float>(m_Widget->m_DoubleSpinBoxMaxRMS->value()));
	m_icp->SetMaxIter(m_Widget->m_SpinBoxMaxIterations->value());
	m_icp->SetApplyPreviousTransform(m_Widget->m_CheckBoxUsePreviousTransformation->isChecked());
	m_icp->SetPreviousTransformMatrix(previousTransformationMatrix);

	// new stuff... strange
	double* bounds = toBeStitched->GetBounds();
	double boundDiagonal = sqrt((bounds[1] - bounds[0])*(bounds[1] - bounds[0]) + (bounds[3] - bounds[2])*(bounds[3] - bounds[2]) + (bounds[5] - bounds[4])*(bounds[5] - bounds[4]));
	m_icp->SetNormalizeRGBToDistanceValuesFactor(static_cast<float>(boundDiagonal / sqrt(3.0)));

	// transform vtkPolyData in our own structures and clip simultaneously
	QTime timeClip;
	timeClip.start();
	m_icp->vtkPolyDataToPointCoordsAndColors(m_Widget->m_DoubleSpinBoxClipPercentage->value());
	CLIP_TIME = timeClip.elapsed();

	QTime timeICP;
	timeICP.start();

	m_icp->Modified();
	m_icp->Update();

	ICP_TIME = timeICP.elapsed();
	m_Widget->m_LabelICPMeanTargetDistance->setText(QString::number(m_icp->GetMeanTargetDistance(), 'f', 2));

	QTime timeTransform;
	timeTransform.start();
	// update output parameter
	outputTransformationMatrix->DeepCopy(m_icp->GetMatrix());

	// perform the transform
	vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter =
		vtkSmartPointer<vtkTransformPolyDataFilter>::New();

	icpTransformFilter->SetInput(toBeStitched);
	icpTransformFilter->SetTransform(m_icp);
	icpTransformFilter->Update();

	outputStitchedPolyData->ShallowCopy(icpTransformFilter->GetOutput());

	TRANSFORM_TIME = timeTransform.elapsed();

	// include the previous transformation into the matrix to allow for "undo"
	if (m_Widget->m_CheckBoxUsePreviousTransformation->isChecked())
	{
		//vtkMatrix4x4::Multiply4x4(outputTransformationMatrix, previousTransformationMatrix, outputTransformationMatrix);
	}

	// update debug information in GUI
	m_Widget->m_LabelICPIterations->setText(QString::number(m_icp->GetNumIter()));
	m_Widget->m_LabelICPError->setText(QString::number(m_icp->GetMeanDist()));
}