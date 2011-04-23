#ifndef VOLE_COMMAND_RADCAL_H
#define VOLE_COMMAND_RADCAL_H

#include <iostream>
#include "command.h"
#include "RadCal/radcal_config.h"

#include "cv.h"

namespace vole {

class RadCal : public Command {
public:
	RadCal();
	~RadCal();
	int execute();

	void printShortHelp() const;
	void printHelp() const;
	void opencvTest();

	RadCalConfig config;

protected:

	std::string getFilePath(std::string filename);

private:

};

}

#endif // VOLE_JPEG_H
