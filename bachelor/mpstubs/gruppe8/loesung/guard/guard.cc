// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/* INCLUDES */

#include "guard/guard.h"
#include "machine/cpu.h"
#include "machine/spinlock.h"

extern CPU cpu; 

void Guard::enter() {
	while (1) {
		cpu.disable_int();
		guard_lock.lock();
		if (!avail()) {
			guard_lock.unlock();
			cpu.enable_int();
		}
		else {
			lock = true;
			guard_lock.unlock();
			cpu.enable_int();
			break;
		}
	}
}

void Guard::leave() {
	leave2();	
	cpu.enable_int();
	
}

bool Guard::leave2() {
	bool epi_Happened = false;
	cpu.disable_int();
	guard_lock.lock();
	Chain *head = epilogue_queue.dequeue();
	Gate *g = (Gate*)head;

	while (g != 0) {
			epi_Happened = true;
			g->queued(false);
			guard_lock.unlock();
			cpu.enable_int();
			g->epilogue();
			cpu.disable_int();
			guard_lock.lock();
			head = epilogue_queue.dequeue();
			g = (Gate *)head;
	} 
	retne();
	guard_lock.unlock();
	return epi_Happened;

}

void Guard::relay(Gate *item) {
	guard_lock.lock();
	if (avail()) {	
		lock = true;
		guard_lock.unlock();
		cpu.enable_int();
		item->epilogue();
		leave();
	} else {
		if (!(item->queued())) {
			item->queued(true);
			epilogue_queue.enqueue((Chain *)item);
		}
		guard_lock.unlock();
	}
}

void Guard::try_idling() {
	if (!leave2()) {
		cpu.idle();
	}
	else {
		cpu.enable_int();
	}
}
