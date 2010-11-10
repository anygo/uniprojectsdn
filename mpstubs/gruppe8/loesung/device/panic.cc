// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/* Hier muesst ihr selbst Code vervollstaendigen */ 
#include "device/panic.h"
#include "machine/cpu.h"
#include "device/cgastr.h"

#include "machine/keyctrl.h"
#include "machine/key.h"


extern CGA_Stream kout;
extern Keyboard_Controller key_controller;

/* Hier muesst ihr selbst Code vervollstaendigen */ 
void Panic::trigger() {
	kout << "panic object wurde aufgerufen - CPU bleibt jetzt stehen. cpu_halt() - NICHT!!!" << endl;
	cpu_halt();
}

bool Panic::prologue() {
	kout << "panic object wurde aufgerufen - CPU bleibt jetzt stehen. cpu_halt()" << endl;
	cpu_halt();

	return false;
}
