// $Date: 2009-11-27 16:35:50 +0100 (Fri, 27 Nov 2009) $, $Revision: 2365 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "syscall/guarded_organizer.h"


void Guarded_Organizer::exit() {
	Secure s;
	Scheduler::exit();
}

void Guarded_Organizer::kill(Thread &thread) {
	Secure s;
	Organizer::kill(thread);
}

void Guarded_Organizer::resume() {
	Secure s;
	Scheduler::resume();
}

void Guarded_Organizer::sleep_until_IRQ() {
	Secure s;
	Scheduler::sleep_until_IRQ();
}
