#ifndef CUDARangeToWorldPLUGIN_H__
#define	CUDARangeToWorldPLUGIN_H__

#include "ApplicationPlugin.h"
#include "CUDARangeToWorldPlugin.h"
#include "CUDARangeToWorldWidget.h"
#include "RImageActorPipeline.h"
#include "RImage.h"

class CUDARangeToWorldPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/// Constructor
	CUDARangeToWorldPlugin();
	/// Destructor
	~CUDARangeToWorldPlugin();

	/// Get the name @sa Plugin::GetName
	QString GetName();

	/// Get the plugins GUI @sa Plugin::GetPluginGUI
	QWidget *GetPluginGUI();

	/// Intercept the event.
	void ProcessEvent(ritk::Event::Pointer EventP);

protected:
	CUDARangeToWorldWidget *m_Widget;	
};


#endif // CUDARangeToWorldPLUGIN_H__
