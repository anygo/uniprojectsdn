// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "meeting/bellringer.h"

void Bellringer::check() {
	bool time_to_party = false;
	Bell *iterator = (Bell *)first();
	while (iterator) {
		iterator->tick();
		if (iterator->run_down()) {
			time_to_party = true;
		}
		iterator = (Bell *)iterator->next;
	}
	if (time_to_party) ring_the_bells();	//YEAH!
}

void Bellringer::job(Bell *bell, int ticks) {
	bell->wait(ticks);
	Bell *iterator = (Bell *)first();
	if (!iterator || iterator->wait() > ticks) {
		insert_first((Chain *) bell);
	}
	else {
		while (iterator->wait() < ticks) {
			if (!iterator->next) {
				insert_after(iterator, bell);
				return;
			}
			iterator = (Bell *)iterator->next;
		}
		insert_after(iterator, bell);
	}
}

void Bellringer::cancel(Bell *bell) {
	remove(bell);
}

void Bellringer::ring_the_bells() {
	Bell *iterator = (Bell *)first();
	while (iterator && iterator->run_down()) {
		iterator->ring();
		Bell *tmp = (Bell *)iterator->next;
		remove(iterator);
		iterator = tmp;
	}
}
