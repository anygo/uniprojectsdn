// $Date: 2009-12-01 16:38:25 +0100 (Tue, 01 Dec 2009) $, $Revision: 2374 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/* INCLUDES */

#include "thread/scheduler.h"
#include "guard/secure.h"
#include "machine/lapic.h"
#include "machine/apicsystem.h"
#include "machine/plugbox.h"
#include "guard/guard.h"
#include "object/debug.h"

extern Guard guard;
extern APICSystem system;
/* IMPLEMENTIERUNG DER METHODEN */

void Scheduler::schedule() {
	Entrant *e = (Entrant *)readyList.dequeue();
	if (!e) {
		go(*idles[system.getCPUID()]);
		return;
	}
	go(*e);
}

void Scheduler::ready(Entrant &that) {
	readyList.enqueue(&that);
	system.sendCustomIPI(0xff, Plugbox::wakeup);
}

void Scheduler::exit() {
	active()->reset_kill_flag();
	Entrant *e = (Entrant *) readyList.dequeue();
	if (!e) {
		dispatch(*idles[system.getCPUID()]);
		return;
	}
	while (e->dying()) {
		e->reset_kill_flag();
		e = (Entrant *) readyList.dequeue();
		if (!e) {
			dispatch(*idles[system.getCPUID()]);
			return;
		}
	}
	dispatch(*e);
}

void Scheduler::kill(Entrant &that) {
	that.set_kill_flag();
	if (readyList.remove(&that)) {
		that.reset_kill_flag();
  	} else {
		system.sendCustomIPI(0xff, Plugbox::assassin);
	}
}

void Scheduler::resume() {
	if (active() != idles[system.getCPUID()]) {
		readyList.enqueue((Entrant *)active());
	}
	Entrant *e = (Entrant *) readyList.dequeue();
	if (!e) {
		dispatch(*idles[system.getCPUID()]);
		return;
	}
	while (e->dying()) {
		e->reset_kill_flag();
		e = (Entrant *) readyList.dequeue();
		if (!e) {
			dispatch(*idles[system.getCPUID()]);
			return;
		}
	}
	dispatch(*e);
}

void Scheduler::sleep_until_IRQ() {
	Entrant *next;
	while ( (next = static_cast<Entrant *>(readyList.dequeue())) == 0) {
		guard.try_idling();
		guard.enter();   // in try_idling wird leave aufgerufen, Zustand wiederherstellen
	}
	dispatch(*next);

}

