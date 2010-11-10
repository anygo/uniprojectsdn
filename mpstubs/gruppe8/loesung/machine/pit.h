// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Enthaelt die Klasse PIT.
 */

#ifndef __pit_include__
#define __pit_include__

#include "machine/io_port.h"

/*! \brief Programmable Interval Timer(PIT)
 * 
 *  Die Klasse PIT steuert den Programmable Interval Timer (PIT) des PCs. 
 */
class PIT {
private:
    PIT(const PIT &copy); // Verhindere Kopieren
	int current_interval;
	IO_Port controlp;
	IO_Port rwport;

public:
	PIT(int us);
	int interval();
	void interval(int us);
};

#endif
