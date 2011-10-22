#include "ReconstructionPlugin.h"
#include "ritkDebugManager.h"
#include "ritkManager.h"
#include "ritkRGBRImage.h"


ReconstructionPlugin::ReconstructionPlugin()
{
	// Create the widget
	m_Widget = new ReconstructionWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget, SLOT(UpdateGUI()));
	//connect(this->m_Widget->m_PushButton, SIGNAL(clicked()), this, SLOT(GetName()));
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

		std::cout << ".";
		ritk::NewFrameEvent::RImagePointer m_ReferenceFrame =
			ritk::NewFrameEvent::RImageType::New();
		std::cout << ".";
		m_ReferenceFrame->DeepCopy(CurrentFrameP);
		std::cout << ".";
		ritk::RGBRImageUCF2::Pointer x = dynamic_cast<ritk::RGBRImageUCF2*>((ritk::RImageF2*) m_ReferenceFrame);
		std::cout << ".";
		std::cout << x->GetRGBImage()->GetBufferPointer()[5][0] << std::endl;
		std::cout << ".";
		emit UpdateGUI();
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
