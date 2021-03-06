#include "ReconstructionPlugin.h"
#include "ritkDebugManager.h"
#include "ritkManager.h"
#include "ritkRGBRImage.h"

#include "ICP.h"
#include "cutil_inline.h"


ReconstructionPlugin::ReconstructionPlugin()
{
	// Create the widget
	m_Widget = new ReconstructionWidget();
	connect(this, SIGNAL(UpdateGUI()), m_Widget, SLOT(UpdateGUI()));

	// DELETE ME ////////////////////////
	// DELETE ME ////////////////////////
	// DELETE ME ////////////////////////

	float** fixed = NULL;
	float** moving = NULL;

	uint maxIter = 50;
	ICP<2048, 6> icp(maxIter); // Initialize ICP

	icp.SetWeight(0.5f); // Set weight for RGB data

	bool devMem = false; // Fixed and moving do not point to GPU memory
	icp.SetFixedPts(fixed, devMem); // Pass fixed point set
	icp.SetMovingPts(moving, devMem); // Pass moving point set

	icp.Run(); // Run ICP
	
	// DELETE ME ////////////////////////
	// DELETE ME ////////////////////////
	// DELETE ME ////////////////////////
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

		size_t freeMemory, totalMemory;
		cudaMemGetInfo(&freeMemory, &totalMemory);
		std::cout << (unsigned long) freeMemory / 1024 / 1024 << " MB / " << (unsigned long) totalMemory / 1024 / 1024 << " MB" << std::endl;

		/*std::cout << ".";
		ritk::NewFrameEvent::RImagePointer m_ReferenceFrame =
			ritk::NewFrameEvent::RImageType::New();
		std::cout << ".";
		m_ReferenceFrame->DeepCopy(CurrentFrameP);
		std::cout << ".";
		ritk::RGBRImageUCF2::Pointer x = dynamic_cast<ritk::RGBRImageUCF2*>((ritk::RImageF2*) m_ReferenceFrame);
		std::cout << ".";
		std::cout << x->GetRGBImage()->GetBufferPointer()[1][0] << std::endl;
		std::cout << ".";
		emit UpdateGUI();*/
	}
	else
	{
		LOG_DEB("Unknown event");
	}
}
