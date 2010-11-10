// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "syscall/guarded_semaphore.h"


void Guarded_Semaphore::wait() {
	Secure s;
	Semaphore::wait();
}

void Guarded_Semaphore::signal() {
	Secure s;
	Semaphore::signal();
}
