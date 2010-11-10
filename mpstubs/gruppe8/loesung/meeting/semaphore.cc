// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "meeting/semaphore.h"
#include "thread/customer.h"
#include "syscall/guarded_organizer.h"

extern Guarded_Organizer organizer;

void Semaphore::wait() {
	if (counter == 0) {
		Customer *c = static_cast<Customer *>(organizer.Organizer::active());
		enqueue(c);
		organizer.Organizer::block(*c, *this);
	}
	else {
		counter--;
	}
}

void Semaphore::signal() {
	Customer *c = static_cast<Customer *>(dequeue());
	if (c) {
		organizer.Organizer::wakeup(*c);
	}
	else {
		counter++;
	}
}
void Semaphore::remove(Customer *customer) {
	Queue::remove(customer);
}
