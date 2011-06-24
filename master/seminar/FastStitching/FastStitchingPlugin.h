#ifndef FastStitchingPLUGIN_H__
#define	FastStitchingPLUGIN_H__

#include "ApplicationPlugin.h"
#include "FastStitchingPlugin.h"
#include "FastStitchingWidget.h"

class FastStitchingPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/// Constructor
	FastStitchingPlugin();
	/// Destructor
	~FastStitchingPlugin();

	/// Get the name @sa Plugin::GetName
	QString GetName();

	/// Get the plugins GUI @sa Plugin::GetPluginGUI
	QWidget *GetPluginGUI();

	/// Intercept the event.
	void ProcessEvent(ritk::Event::Pointer EventP);

signals:
	void UpdateGUI();

protected:
	FastStitchingWidget *m_Widget;	
};


#endif // FastStitchingPLUGIN_H__
