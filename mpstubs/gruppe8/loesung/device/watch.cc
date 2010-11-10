// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "device/watch.h"
#include "machine/plugbox.h"
#include "syscall/guarded_organizer.h"
#include "machine/ioapic.h"
#include "guard/guard.h"
#include "device/cgastr.h"
#include "object/debug.h"
#include "meeting/bellringer.h"
#include "syscall/guarded_vesagraphics.h"


extern CGA_Stream kout;
extern Plugbox plugbox;
extern IOAPIC ioapic;
extern Guarded_Organizer organizer;
extern Guard guard;
extern CGA_Stream dall;
extern Bellringer bellringer;
extern Guarded_VESAGraphics vesa;


void Watch::windup() {
	plugbox.assign(Plugbox::timer, *this);
	ioapic.allow(IOAPIC::timer);
}

bool Watch::prologue() {
	return true;
}

void Watch::epilogue() {
	
//	vesa.VESAGraphics::switch_buffers();
//	vesa.VESAGraphics::scanout_frontbuffer();

	
	bellringer.check();

	organizer.Organizer::resume();
}
