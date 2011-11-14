#ifndef RECONSTRUCTIONWIDGET_H__
#define RECONSTRUCTIONWIDGET_H__

#include <QWidget>
#include "Ui_ReconstructionWidget.h"

#include "ritkRImage.h"

class ReconstructionWidget : public QWidget, public Ui_ReconstructionWidget
{
	Q_OBJECT

public:
	/// Constructor
	ReconstructionWidget();
	/// Destructor
	~ReconstructionWidget();


		/** @name Typedefs for the range image and components*/
	//@{
	typedef ritk::RImageF2 RImageType;
	typedef RImageType::Pointer RImagePointer;
	typedef RImageType::ConstPointer RImageConstPointer;
	//@}

public slots:
	void UpdateGUI();
};

#endif // RECONSTRUCTIONWIDGET_H__
