// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/* INCLUDES */

#include "machine/plugbox.h"
#include "device/panic.h"
/* Hier muesst ihr selbst Code vervollstaendigen */ 

extern Panic panic;

Plugbox::Plugbox() {
	for (int i = 0; i < 128; i++)
	{
		gate_map[i] = &panic;
	}
}

void Plugbox::assign(unsigned int slot, Gate &gate) {
	gate_map[slot] = &gate;

}

Gate &Plugbox::report(unsigned int slot) {
	return *(gate_map[slot]);
}
