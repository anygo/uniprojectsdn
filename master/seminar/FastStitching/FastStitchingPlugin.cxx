#include "CUDARangeToWorldPlugin.h"
#include "DebugManager.h"
#include "Manager.h"


CUDARangeToWorldPlugin::CUDARangeToWorldPlugin()
{
	// Create the widget
	m_Widget = new CUDARangeToWorldWidget();

	Timer::SetNumberOfPassesToAverageRuntime(10);
}

CUDARangeToWorldPlugin::~CUDARangeToWorldPlugin()
{
	delete m_Widget;
}


//----------------------------------------------------------------------------
QString
CUDARangeToWorldPlugin::GetName()
{
	return tr("CUDARangeToWorldPlugin");
}


//----------------------------------------------------------------------------
QWidget*
CUDARangeToWorldPlugin::GetPluginGUI()
{
	return m_Widget;
}


//----------------------------------------------------------------------------
void
CUDARangeToWorldPlugin::ProcessEvent(ritk::Event::Pointer EventP)
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
		Timer::StartTimer();
		m_Widget->SetRangeData(CurrentFrameP);
		Timer::StopTimer();
		Timer::GetElapsedTime(&m_Runtime[0],&m_Runtime[1]);
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
