// $Date: 2009-12-09 16:35:41 +0100 (Wed, 09 Dec 2009) $, $Revision: 2393 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "meeting/buzzer.h"
#include "syscall/guarded_organizer.h"

extern Guarded_Organizer organizer;

Buzzer::~Buzzer() {
	bellringer.cancel(this);
	ring();
}

void Buzzer::ring() {
	Customer *c;
	while ((c = static_cast<Customer *>(dequeue()))) {
		organizer.Organizer::wakeup(*c);
	}
}

void Buzzer::set(int ms) {
	counter = ms;
}

void Buzzer::sleep() {
	bellringer.job(this, counter);
	Customer *c = static_cast<Customer *>(organizer.Organizer::active());
	enqueue(c);
	organizer.Organizer::block(*c, *this);
}

void Buzzer::sleep(int ms) {
	set(ms);
	sleep();
}
