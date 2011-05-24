#ifndef StitchingWIDGET_H__
#define StitchingWIDGET_H__

#include <QWidget>
#include "Ui_StitchingWidget.h"

#include "RImage.h"

class StitchingWidget : public QWidget, public Ui_StitchingWidget
{
	Q_OBJECT

public:
	/// Constructor
	StitchingWidget();
	/// Destructor
	~StitchingWidget();


	/** @name Typedefs for the range image and components*/
	//@{
	typedef ritk::RImageF2 RImageType;
	typedef RImageType::Pointer RImagePointer;
	typedef RImageType::ConstPointer RImageConstPointer;
	//@}

public slots:
	void UpdateGUI();

};

#endif // StitchingWIDGET_H__
