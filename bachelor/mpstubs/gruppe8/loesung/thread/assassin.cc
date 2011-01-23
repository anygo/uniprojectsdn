// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "thread/assassin.h"
#include "machine/plugbox.h"
#include "machine/ioapic.h"
#include "syscall/guarded_organizer.h"
#include "machine/lapic.h"
#include "device/cgastr.h"
#include "machine/apicsystem.h"
#include "object/debug.h"
#include "machine/cpu.h"
#include "guard/guard.h"

extern Plugbox plugbox;
extern IOAPIC ioapic;
extern Guarded_Organizer organizer;
extern LAPIC lapic;
extern APICSystem system;
extern CPU cpu;
extern Guard guard;

void Assassin::hire() {
	plugbox.assign(Plugbox::assassin, *this);
}

bool Assassin::prologue() {
	int a = organizer.active() && organizer.active()->dying();
	DBG << "assassin!!! " << a << endl;
	
	return a;
}

void Assassin::epilogue() {
	if (organizer.active() && organizer.active()->dying()) {
		DBG << "Active thread died..." << endl;
		organizer.Organizer::exit(); 
	}
}
