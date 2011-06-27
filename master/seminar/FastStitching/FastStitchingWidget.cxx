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

	m_CurrentFrame = NULL;

	for (int i = 0; i < BUFFERED_FRAMES; ++i)
	{
		m_RImage[i] = ritk::OpenGLRImageEntity::New();
		//m_VisualizationUnit->AddEntity(m_RImage[i]);
	}

	m_VisualizationUnit->update();
	m_VisualizationUnit->show();
}

FastStitchingWidget::~FastStitchingWidget()
{
}

void
FastStitchingWidget::ShowFrame(float4* stitched)
{
	std::cout << "ShowFrame()" << std::endl;

	m_Mutex.lock();

	ritk::RImageF2::Pointer ptr;
	ptr = ritk::RImageF2::New();
	ptr->DeepCopy(m_CurrentFrame);

	QTime t;
	t.start();

	ritk::RImageF2::WorldCoordType* bufPtr = ptr->GetWorldCoordImage()->GetBufferPointer();
	for (int i = 0; i < 640*480; ++i)
	{
		ritk::RImageF2::WorldCoordType* buf = &bufPtr[i];
		buf->GetDataPointer()[0] = stitched[i].x;
		buf->GetDataPointer()[1] = stitched[i].y;
		buf->GetDataPointer()[2] = stitched[i].z;
	}
	std::cout << "umwandeln: " << t.elapsed() << std::endl;

	static int ctr = 0;
	m_RImage[ctr%BUFFERED_FRAMES]->SetRangeData((ritk::RImageF2::ConstPointer) ptr);
	m_VisualizationUnit->AddEntity(m_RImage[ctr%BUFFERED_FRAMES]);
	m_VisualizationUnit->update();
	m_VisualizationUnit->show();

	++ctr;

	m_Mutex.unlock();
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
	//m_RImage->SetRangeData(m_CurrentFrame);

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
