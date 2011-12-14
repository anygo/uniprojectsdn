#ifndef FASTICPWIDGET_H__
#define FASTICPWIDGET_H__

#include <QWidget>
#include "ui_FastICPWidget.h"

#include "ritkRImage.h"

class FastICPWidget : public QWidget, public Ui_FastICPWidget
{
	Q_OBJECT

public:
	/// Constructor
	FastICPWidget();
	/// Destructor
	~FastICPWidget();


		/** @name Typedefs for the range image and components*/
	//@{
	typedef ritk::RImageF2 RImageType;
	typedef RImageType::Pointer RImagePointer;
	typedef RImageType::ConstPointer RImageConstPointer;
	//@}

public slots:
	void UpdateGUI();

};

#endif // FASTICPWIDGET_H__
