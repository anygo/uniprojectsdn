// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "machine/key.h"
#include "device/keyboard.h"
#include "machine/ioapic.h"
#include "guard/gate.h"
#include "machine/plugbox.h"
#include "device/cgastr.h"
#include "machine/keyctrl.h"
#include "machine/spinlock.h"
#include "guard/secure.h"
#include "object/debug.h"
#include "user/appl.h"

extern IOAPIC ioapic;
extern CGA_Stream kout;
extern Plugbox plugbox;
extern Keyboard_Controller key_controller;
extern Guard guard;

void Keyboard::plugin() {
	ioapic.config(IOAPIC::keyboard, Plugbox::keyboard);
	plugbox.assign(Plugbox::keyboard, *this);
	ioapic.allow(IOAPIC::keyboard);
}

bool Keyboard::prologue() {
		
		currentkey = key_controller.key_hit();

		if (currentkey.ctrl() && currentkey.alt() && (currentkey.scancode() == 0x53)) {
			reboot();
		}

		if (currentkey.valid()) {
			return true;
		}

		return false;
}

Key Keyboard::getkey() {
	sema.wait();
	return puf;
}

void Keyboard::epilogue() {
	if (currentkey.valid()) {
		puf = currentkey;
		sema.Semaphore::signal();		
	}
}
