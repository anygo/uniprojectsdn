#ifndef IMPROVEDSTITCHINGWIDGET_H__
#define IMPROVEDSTITCHINGWIDGET_H__

#include <QWidget>
#include "ui_ImprovedStitchingWidget.h"

#include "ritkRImage.h"

class ImprovedStitchingWidget : public QWidget, public Ui_ImprovedStitchingWidget
{
	Q_OBJECT

public:
	/// Constructor
	ImprovedStitchingWidget();
	/// Destructor
	~ImprovedStitchingWidget();


		/** @name Typedefs for the range image and components*/
	//@{
	typedef ritk::RImageF2 RImageType;
	typedef RImageType::Pointer RImagePointer;
	typedef RImageType::ConstPointer RImageConstPointer;
	//@}

public slots:
	void UpdateGUI();

};

#endif // IMPROVEDSTITCHINGWIDGET_H__
