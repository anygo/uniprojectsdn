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
	connect(this,									SIGNAL(NewFrameToStitch()),			m_Stitcher, SLOT(Stitch())			);
	connect(this->m_Stitcher,			SIGNAL(FrameStitched(float4*)),		this,	SLOT(ShowFrame(float4*))				);

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
	std::cout << "umwandeln: " << t.elapsed() << " ms" << std::endl;

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
	m_CurrentFrame = Data;

	m_Stitcher->Prepare(m_CurrentFrame);
	//m_RImage->SetRangeData(m_CurrentFrame);


	static int counter;
	if (++counter > 1)
		emit NewFrameToStitch();
}

