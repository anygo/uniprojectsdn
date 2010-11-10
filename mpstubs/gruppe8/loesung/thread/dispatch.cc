// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "thread/dispatch.h"

extern APICSystem system;

void Dispatcher::go(Coroutine &first) {
	life[(int)system.getCPUID()] = &first;
	first.go();
}

void Dispatcher::dispatch(Coroutine &next) {
	Coroutine *tmp = life[(int)system.getCPUID()];
	life[(int)system.getCPUID()] = &next;
	tmp->resume(next);
}

Coroutine * Dispatcher::active() {
	return life[(int)system.getCPUID()];
}
