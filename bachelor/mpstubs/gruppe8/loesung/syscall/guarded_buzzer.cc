// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "syscall/guarded_buzzer.h"
#include "guard/secure.h"

Guarded_Buzzer::~Guarded_Buzzer() {
	Secure s;
}

void Guarded_Buzzer::set(int ms) {
	Secure s;
	Buzzer::set(ms);
}

void Guarded_Buzzer::sleep() {
	Secure s;
	Buzzer::sleep();
}
