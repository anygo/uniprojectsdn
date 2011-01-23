// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "meeting/waitingroom.h"
#include "syscall/guarded_organizer.h"

extern Guarded_Organizer organizer;

Waitingroom::~Waitingroom() {
	Customer *c;
	while ((c = ((Customer *)dequeue())))  {	
		organizer.Organizer::wakeup(*c);
	}	
}

void Waitingroom::remove(Customer *customer) {
	Queue::remove((Chain *) customer);
}
