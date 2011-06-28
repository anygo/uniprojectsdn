#ifndef FASTSTITCHINGWIDGET_H__
#define FASTSTITCHINGWIDGET_H__

#include "RITKVisualization.h"

#include <QtOpenGL>
#include <QGLWidget>

#include <QMutex>

#include "Ui_FastStitchingWidget.h"

#include "RImage.h"
#include "RImageActorPipeline.h"

#include "VisualizationUnit.h"

#define BUFFERED_FRAMES 5

class FastStitchingWidget : public QWidget, public Ui_FastStitchingWidget
{
	Q_OBJECT

public:
	/// Constructor
	FastStitchingWidget(QWidget *parent=0);
	/// Destructor
	~FastStitchingWidget();

public slots:
	void SetRangeData(ritk::RImageF2::ConstPointer Data);
	void ShowFrame(float4*);

//protected slots:

signals:
	void NewFrameToStitch();

protected:
	/// Mutex used to synchronize rendering and data update
	QMutex m_Mutex;

	/// The current frame
	ritk::RImageF2::ConstPointer m_CurrentFrame;

	ritk::OpenGLRImageEntity::Pointer m_RImage[BUFFERED_FRAMES];
};

#endif // FASTSTITCHINGWIDGET_H__
