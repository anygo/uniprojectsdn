#ifndef VOLE_COMMAND_GEOINV_H
#define VOLE_COMMAND_GEOINV_H

#include <iostream>
#include "command.h"
#include "GeoInv/geoinv_config.h"


namespace vole {

class GeoInv : public Command {
public:
	GeoInv();
	~GeoInv();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

	GeoInvConfig config;

protected:

private:

};

}

#endif // VOLE_COMMAND_GEOINV_H
