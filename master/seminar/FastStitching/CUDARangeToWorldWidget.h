#ifndef CUDARANGETOWORLDWIDGET_H__
#define CUDARANGETOWORLDWIDGET_H__

#include "RITKVisualization.h"

#include <QtOpenGL>
#include <QGLWidget>

#include <QMutex>

#include "Ui_CUDARangeToWorldWidget.h"

#include "RImage.h"

class CUDARangeToWorldWidget : public QWidget, public Ui_CUDARangeToWorldWidget
{
	Q_OBJECT

public:
	/// Constructor
	CUDARangeToWorldWidget(QWidget *parent=0);
	/// Destructor
	~CUDARangeToWorldWidget();

public slots:
	/**	@brief	Set the TOF data to render 
	 *	@param	Data	The data to render
	 *
	 *	@details
	 *	A call to this method will set the TOF data to render.
	 *	In order to comply with the OpenGL render context this method
	 *	will pass the data to the UpdateVBO() slot via the NewDataAvailable() signal.
	 */
  void SetRangeData(ritk::RImageF2::ConstPointer Data);

protected slots:
	/// Connected to the alpha slider. Delegates the normalized value to the OpenGL widget
	void LUTAlphaSliderMoved(int value);
	/// Connected to the LUT combo box. Delegates the index to the OpenGL widget
	void LUTIndexChanged(int index);
  /// Connected to the Radiobuttons
	void RadioButtonPolyDataClicked();

	// Connected to the widget's range interval min spinbox
	void RangeIntervalMinChanged(double d);
	// Connected to the widget's range interval max spinbox
	void RangeIntervalMaxChanged(double d);
	// Connected to the widget's clamp range interval button
	void ClampRangeInterval();

  void SetMinValue(int value);
  void SetMaxValue(int value);
signals:
  void SetMinSignal(int value);
  void SetMaxSignal(int value);

protected:
	/// Mutex used to synchronize rendering and data update
	QMutex m_Mutex;

	/// The current frame
  ritk::RImageF2::ConstPointer m_CurrentFrame;
};

#endif // CUDARANGETOWORLDWIDGET_H__
