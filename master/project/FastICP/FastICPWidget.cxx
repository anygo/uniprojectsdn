#include "FastICPWidget.h"
#include "ritkDebugManager.h"

#include <windows.h> 


//----------------------------------------------------------------------------
FastICPWidget::FastICPWidget()
{
	setupUi(this);
}


//----------------------------------------------------------------------------
FastICPWidget::~FastICPWidget()
{
}


//----------------------------------------------------------------------------
void
FastICPWidget::UpdateGUI()
{	
	m_VisualizationWidget3D->UpdateGUI();
	m_VisualizationWidget3DVolume->UpdateGUI();
}