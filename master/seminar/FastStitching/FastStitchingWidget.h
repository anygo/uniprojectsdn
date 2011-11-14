#ifndef FastStitchingWIDGET_H__
#define FastStitchingWIDGET_H__

#include <QWidget>
#include "Ui_FastStitchingWidget.h"

#include "ritkRImage.h"

class FastStitchingWidget : public QWidget, public Ui_FastStitchingWidget
{
	Q_OBJECT

public:
	/// Constructor
	FastStitchingWidget();
	/// Destructor
	~FastStitchingWidget();


	/** @name Typedefs for the range image and components*/
	//@{
	typedef ritk::RImageF2 RImageType;
	typedef RImageType::Pointer RImagePointer;
	typedef RImageType::ConstPointer RImageConstPointer;
	//@}

public slots:
	void UpdateGUI();

};

#endif // FastStitchingWIDGET_H__
