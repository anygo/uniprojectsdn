// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "thread/organizer.h"

void Organizer::block(Customer &customer, Waitingroom &waitingroom) {
	customer.waiting_in(&waitingroom);
	Scheduler::exit();
}

void Organizer::wakeup(Customer &customer) {
	customer.waiting_in(0);
	ready(customer);	
}

void Organizer::kill(Customer &customer) {
	Waitingroom *w;
	if ((w = customer.waiting_in())) {
		w->remove(&customer);
	}
		
	Scheduler::kill(customer);
}
