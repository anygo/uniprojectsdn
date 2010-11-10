// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "machine/toc.h"

// TOC_SETTLE: bereitet den Kontext der Koroutine fuer den ersten
//             Aufruf vor.
void toc_settle (struct toc* regs, void* tos, void (*kickoff)(Coroutine*),
        Coroutine* object) {
	
	void **sp = (void **)tos;
	*(--sp) = object;
	*(--sp) = 0;			// hier wuerde Ruecksprungadresse stehen, aber Coroutine soll nie zurueckkehren
	*(--sp) = (void *)kickoff;
	regs->esp = sp;
}
