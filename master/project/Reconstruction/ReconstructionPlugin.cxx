#include "ReconstructionPlugin.h"
#include "ritkDebugManager.h"
#include "ritkManager.h"


ReconstructionPlugin::ReconstructionPlugin()
{
	// Create the widget
	m_Widget = new ReconstructionWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget, SLOT(UpdateGUI()));
}

ReconstructionPlugin::~ReconstructionPlugin()
{
	delete m_Widget;
}


//----------------------------------------------------------------------------
QString
ReconstructionPlugin::GetName()
{
	return tr("ReconstructionPlugin");
}


//----------------------------------------------------------------------------
QWidget*
ReconstructionPlugin::GetPluginGUI()
{
	return m_Widget;
}


//----------------------------------------------------------------------------
void
ReconstructionPlugin::ProcessEvent(ritk::Event::Pointer EventP)
{
	// New frame event
	if ( EventP->type() == ritk::NewFrameEvent::EventType )
	{
		// Cast the event
		ritk::NewFrameEvent::Pointer NewFrameEventP = qSharedPointerDynamicCast<ritk::NewFrameEvent,ritk::Event>(EventP);
		if ( !NewFrameEventP )
		{
			LOG_DEB("Event mismatch detected: Type=" << EventP->type());
			return;
		}
		ritk::NewFrameEvent::RImageConstPointer CurrentFrameP = NewFrameEventP->RImage;

		// Here comes your code. Access range data with CurrentFrame.
		// ...

		std::cout << "HELLOOOOOOOO" << std::endl;

		emit UpdateGUI();
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
