		// $Date: 2009-12-01 16:38:25 +0100 (Tue, 01 Dec 2009) $, $Revision: 2374 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __schedule_include__
#define __schedule_include__

/*! \file
 * 
 *  \brief Enthaelt die Klasse Scheduler
 */

#include "thread/dispatch.h"
#include "object/queue.h"
#include "thread/entrant.h"
#include "syscall/thread.h"

/*! \brief Implementierung des Schedulers.
 *  
 *  Der Scheduler verwaltet die Ready-Liste (ein privates Queue Objekt der 
 *  Klasse), also die Liste der lauffaehigen Prozesse (Entrant Objekte). Die 
 *  Liste wird von vorne nach hinten abgearbeitet. Dabei werden Prozesse, die 
 *  neu im System sind oder den Prozessor abgeben, stets an das Ende der Liste 
 *  angefuegt. 
 */ 
class Scheduler : public Dispatcher
{
private:
    Scheduler (const Scheduler &copy); // Verhindere Kopieren
	Thread *idles[CPU_MAX];

protected: 
	Queue readyList; 

public:
	Scheduler() {for(int i = 0; i < CPU_MAX; i++) idles[i] = 0;} 
	void schedule();
	void ready(Entrant &that);
	void exit();
	void kill(Entrant &that);
	void resume();
	inline void set_idle_thread(int cpuid, Entrant *thread) {idles[cpuid] = (Thread *)thread;}
	void sleep_until_IRQ();
};

#endif
