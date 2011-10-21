#ifndef ReconstructionPLUGIN_H__
#define	ReconstructionPLUGIN_H__

#include "ritkApplicationPlugin.h"
#include "ReconstructionPlugin.h"
#include "ReconstructionWidget.h"

class ReconstructionPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/// Constructor
	ReconstructionPlugin();
	/// Destructor
	~ReconstructionPlugin();

	/// Get the name @sa Plugin::GetName
	QString GetName();

	/// Get the plugins GUI @sa Plugin::GetPluginGUI
	QWidget *GetPluginGUI();

	/// Intercept the event.
	void ProcessEvent(ritk::Event::Pointer EventP);

signals:
	void UpdateGUI();

protected:
	ReconstructionWidget *m_Widget;	
};


#endif // ReconstructionPLUGIN_H__
