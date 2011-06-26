#include "FastStitchingWidget.h"

#include <vtkActor.h>
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

#include "RImage.h"
#include "RImageActorPipeline.h"

FastStitchingWidget::FastStitchingWidget(QWidget *parent) :
QWidget(parent)
{
	setupUi(this);

	// Connect signals and slots
	connect(this->m_AlphaSlider,					SIGNAL(sliderMoved(int)),			this,	SLOT(LUTAlphaSliderMoved(int))			);
	connect(this->m_LUTComboBox,					SIGNAL(currentIndexChanged(int)),	this,	SLOT(LUTIndexChanged(int))				);
	connect(this->m_RadioButtonPoints,				SIGNAL(clicked()),					this,	SLOT(RadioButtonPolyDataClicked())		);
	connect(this->m_RadioButtonTriangles,			SIGNAL(clicked()),					this,	SLOT(RadioButtonPolyDataClicked())		);
	connect(this->m_RangeIntervalMinSpinBox,		SIGNAL(valueChanged(double)),		this,	SLOT(RangeIntervalMinChanged(double))	);
	connect(this->m_RangeIntervalMaxSpinBox,		SIGNAL(valueChanged(double)),		this,	SLOT(RangeIntervalMaxChanged(double))	);
	connect(this->m_RangeIntervalClampPushButton,	SIGNAL(clicked()),					this,	SLOT(ClampRangeInterval())				);
	connect(this,									SIGNAL(SetMinSignal(int)),			this,	SLOT(SetMinValue(int))					);
	connect(this,									SIGNAL(SetMaxSignal(int)),			this,	SLOT(SetMaxValue(int))					);
	connect(this,									SIGNAL(NewFrameToStitch()),			m_VisualizationWidget3D, SLOT(Stitch())			);
	connect(this->m_VisualizationWidget3D,			SIGNAL(FrameStitched(float4*)),		this,	SLOT(ShowFrame(float4*))				);
	connect(this, 									SIGNAL(NewPolyDataAvailable()),		m_VisualizationWidget3D_VTK, SLOT(UpdateGUI())	);

	m_CurrentFrame = NULL;

	m_DataActor3D = vtkSmartPointer<ritk::RImageActorPipeline>::New();
}

FastStitchingWidget::~FastStitchingWidget()
{
}

void
FastStitchingWidget::ShowFrame(float4* stitched)
{
	std::cout << "hallo" << std::endl;

	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> cells =
		vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkDataArray> colors =
		vtkSmartPointer<vtkUnsignedCharArray>::New();
	vtkSmartPointer<vtkPolyData> data =
		vtkSmartPointer<vtkPolyData>::New();

	colors->SetNumberOfComponents(4);

	float4 p;
	for (int i = 0; i < 640*480; i += 8)
	{
		p = stitched[i];

		if (p.x == p.x) // i.e. not QNAN
		{
			double colorTuple[4] = { 255, 255, 255, 255};
			points->InsertNextPoint(p.x, p.y, p.z);
			colors->InsertNextTuple(colorTuple);
		}
	}

	for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
	{
		cells->InsertNextCell(1, &i);
	}	

	// update m_Data
	data->SetPoints(points);
	data->SetVerts(cells);
	data->GetPointData()->SetScalars(colors);
	data->Modified();
	data->Update();

		
	m_DataActor3D->SetVisualizationMode(ritk::RImageActorPipeline::RGB);

	/*m_DataActor3D->SetData(m_CurrentFrame);
	m_DataActor3D->Modified();*/
	//m_VisualizationWidget3D_VTK->AddActor(m_DataActor3D);

	m_DataActor3D->SetData(data, true);
	m_DataActor3D->Modified();

	std::cout << data->GetPoints()->GetNumberOfPoints() << " points" << std::endl;

	m_VisualizationWidget3D_VTK->AddActor(m_DataActor3D);


	// write to file
	/*static int counter = 0;
	std::stringstream ss;
	ss << "file_" << ++counter << ".vtk";
	vtkSmartPointer<vtkPolyDataWriter> writer =
		vtkSmartPointer<vtkPolyDataWriter>::New();
	writer->SetFileName(ss.str().c_str());
	writer->SetInput(data);
	writer->SetFileTypeToASCII();
	writer->Update();*/
	
	emit NewPolyDataAvailable();
}

//----------------------------------------------------------------------------
void 
FastStitchingWidget::SetRangeData(ritk::RImageF2::ConstPointer Data)
{
	if ( !m_CurrentFrame )
	{
		m_CurrentFrame = Data;

		this->m_RangeIntervalMinSpinBox->setEnabled(true);
		this->m_RangeIntervalMaxSpinBox->setEnabled(true);
		this->m_RangeIntervalClampPushButton->setEnabled(true);

		ClampRangeInterval();
	}
	m_CurrentFrame = Data;

	m_VisualizationWidget3D->SetRangeData(m_CurrentFrame);

	static int counter;
	if (++counter > 1)
		emit NewFrameToStitch();
}


//----------------------------------------------------------------------------
void 
FastStitchingWidget::SetMinValue(int value)
{
	m_RangeIntervalMinSpinBox->setValue(value);
}
//----------------------------------------------------------------------------
void 
FastStitchingWidget::SetMaxValue(int value)
{
	m_RangeIntervalMaxSpinBox->setValue(value);
}


//----------------------------------------------------------------------------
void 
FastStitchingWidget::LUTAlphaSliderMoved(int value)
{
	m_VisualizationWidget3D->SetLUTAlpha(value/100.f);
}


//----------------------------------------------------------------------------
void 
FastStitchingWidget::LUTIndexChanged(int index)
{
	m_VisualizationWidget3D->SetLUT(index);
}


//----------------------------------------------------------------------------
void 
FastStitchingWidget::RadioButtonPolyDataClicked()
{
	if(this->m_RadioButtonPoints->isChecked())
	{
		m_VisualizationWidget3D->SetRenderType(true);
	}
	else if(this->m_RadioButtonTriangles->isChecked())
	{
		m_VisualizationWidget3D->SetRenderType(false);
	}
}


//----------------------------------------------------------------------------
void
FastStitchingWidget::RangeIntervalMinChanged(double d)
{
	if ( d >= this->m_RangeIntervalMaxSpinBox->value() )
		return;

	m_VisualizationWidget3D->SetRangeClamping(d, this->m_RangeIntervalMaxSpinBox->value());
}


//----------------------------------------------------------------------------
void
FastStitchingWidget::RangeIntervalMaxChanged(double d)
{
	if ( d <= this->m_RangeIntervalMinSpinBox->value() )
		return;

	m_VisualizationWidget3D->SetRangeClamping(this->m_RangeIntervalMinSpinBox->value(), d);
}


//----------------------------------------------------------------------------
void
FastStitchingWidget::ClampRangeInterval()
{
	m_Mutex.lock();
	// Get the min and max value from the current frame
	float RMin = 1e16;
	float RMax = -1e16;
	if ( m_CurrentFrame )
	{
		// For convenience
		long SizeX = m_CurrentFrame->GetBufferedRegion().GetSize()[0];
		long SizeY = m_CurrentFrame->GetBufferedRegion().GetSize()[1];

		const float *RData = m_CurrentFrame->GetRangeImage()->GetBufferPointer();

		for ( long l = 0; l < SizeX*SizeY; l++ )
		{
			if ( fabs(RData[l]) < 1e-5 )
				continue;
			if ( RData[l] < RMin )
				RMin = RData[l];
			if ( RData[l] > RMax )
				RMax = RData[l];
		}	
	}

	m_Mutex.unlock();

	m_VisualizationWidget3D->SetRangeClamping(RMin,RMax);

	// needs to be done with slot-signal because of contexts
	emit SetMinSignal(RMin);
	emit SetMaxSignal(RMax);
}
