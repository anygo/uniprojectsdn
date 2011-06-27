#include "FastStitchingPlugin.h"
#include "DebugManager.h"
#include "Manager.h"


FastStitchingPlugin::FastStitchingPlugin()
{
	// Create the widget
	m_Widget = new FastStitchingWidget();

	Timer::SetNumberOfPassesToAverageRuntime(10);
}

FastStitchingPlugin::~FastStitchingPlugin()
{
	delete m_Widget;
}


//----------------------------------------------------------------------------
QString
FastStitchingPlugin::GetName()
{
	return tr("FastStitchingPlugin");
}


//----------------------------------------------------------------------------
QWidget*
FastStitchingPlugin::GetPluginGUI()
{
	return m_Widget;
}


//----------------------------------------------------------------------------
void
FastStitchingPlugin::ProcessEvent(ritk::Event::Pointer EventP)
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

		static int counter = 0;
		Timer::StartTimer();
		if (++counter % m_Widget->m_SpinBoxSkipFrames->value() == 0)
			m_Widget->SetRangeData(CurrentFrameP);

		Timer::StopTimer();
		Timer::GetElapsedTime(&m_Runtime[0],&m_Runtime[1]);
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
