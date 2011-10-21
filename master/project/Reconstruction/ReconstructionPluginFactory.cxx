#include "ReconstructionPluginFactory.h"
#include "ReconstructionPlugin.h"

#include <QtPlugin>

ReconstructionPluginFactory::ReconstructionPluginFactory()
{
}

ReconstructionPluginFactory::~ReconstructionPluginFactory()
{
}

ritk::ApplicationPlugin*
ReconstructionPluginFactory::CreateInstance()
{
	return new ReconstructionPlugin();
}

QString 
ReconstructionPluginFactory::GetName()
{
	return ("Reconstruction");
}

QString 
ReconstructionPluginFactory::GetDescription()
{
	return ("Real-time environment reconstruction");
}


Q_EXPORT_PLUGIN2(ReconstructionPlugin, ReconstructionPluginFactory)

