// $Date: 2009-09-25 16:32:11 +0200 (Fri, 25 Sep 2009) $, $Revision: 2224 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Low-Level Interrupt-Behandlung
 */

/* INCLUDES */
#include "object/debug.h"
#include "machine/plugbox.h"
#include "machine/lapic.h"
#include "machine/cpu.h"
#include "machine/keyctrl.h"
#include "guard/guard.h"

extern Keyboard_Controller::Keyboard_Controller keyboard;
extern CGA_Stream::CGA_Stream kout;
extern Plugbox plugbox;
extern LAPIC lapic;
extern CPU cpu;
extern Guard guard;
#include "object/debug.h"

/* FUNKTIONEN */

extern "C" void guardian(unsigned int slot);


/*! \brief Low-Level Interrupt-Behandlung.
 *  
 *  Zentrale Unterbrechungsbehandlungsroutine des Systems. 
 *  \param slot gibt die Nummer des aufgetretenen Interrupts an.
 */
void guardian(unsigned int slot) {
	//DBG << slot << endl;
	Gate *g = &(plugbox.report(slot));
	bool b = g->prologue();
	lapic.ackIRQ();
	if (b) {
		guard.relay(g);
	}
}
