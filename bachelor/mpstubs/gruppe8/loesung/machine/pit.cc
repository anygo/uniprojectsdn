// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "machine/pit.h"


PIT::PIT(int us) : controlp(0x43), rwport(0x40) {
	interval(us);
}

void PIT::interval(int us) {
	current_interval = us;

	// Zaehler#  erst Low, dann High  periodischer Modus  Zaehlart: binaer
	// 00        11                   010                 0
	//
	// -> 00110100 = 0x34
	controlp.outb(0x34);

	// Umrechnung in us wegen 838ns-quatsch ;)
	us *= 1000; // -> ns
	us /= 838; // 838ns-quatsch!!!

	unsigned char low = us & 0x00ff;
	unsigned char high = 0; 
	high = us >> 8;

	rwport.outb(low);
	rwport.outb(high); 
}

int PIT::interval() {
	return current_interval;
}
