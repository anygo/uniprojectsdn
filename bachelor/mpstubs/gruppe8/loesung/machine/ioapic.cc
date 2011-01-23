// $Date: 2009-08-14 16:19:58 +0200 (Fri, 14 Aug 2009) $, $Revision: 2209 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "ioapic.h"

extern APICSystem system;

void IOAPIC::init() {

	IOAPICRegister_t ioar_id;
	IOAPICRegister_t ioar_r_l;
	IOAPICRegister_t ioar_r_h;
	ioar_id.IOAPICID.ID = system.getIOAPICID();

	IOREGSEL_REG = 0x00;
	IOWIN_REG = ioar_id.value;

	struct IOREDTBL_L low;
	struct IOREDTBL_H high;
	low.vector = 32; // Panic hoffentlich 
	low.delivery_mode = DELIVERY_MODE_LOWESTPRI;
	low.destination_mode = DESTINATION_MODE_LOGICAL;
	low.delivery_status = 0;
	low.polarity = POLARITY_HIGH;
	low.remote_irr = 0;
	low.trigger_mode = TRIGGER_MODE_EDGE;
	low.mask = MASK_DISABLED;
	low.reserved = 0;
	high.logical_destination = 0xff; // alle CPUs
	high.reserved = 0;

	ioar_r_l.IOREDTBL_L = low;
	ioar_r_h.IOREDTBL_H = high;

	for (int i = 0; i < 24; i++) {
		IOREGSEL_REG = 0x10 + i*0x02;
		
		IOWIN_REG = ioar_r_l.value;
		IOREGSEL_REG = (0x10 + i*0x02) + 0x01;
		IOWIN_REG = ioar_r_h.value;
	}

}

void IOAPIC::config(IOAPIC::Slot slot, Plugbox::Vector vector) {
	IOREGSEL_REG = 0x10 + slot*0x02;
	IOWIN_REG &= 0xffffff00;					//Vektoreintraege auf 0 setzen
	IOWIN_REG |= vector;
}

void IOAPIC::forbid(IOAPIC::Slot interrupt) {
	IOREGSEL_REG = 0x10 + interrupt*0x02;
	IOWIN_REG |= 0x00010000; // 16. bit auf 1 setzen -> ausmaskiert
}

void IOAPIC::allow(IOAPIC::Slot interrupt) {
	IOREGSEL_REG = 0x10 + interrupt*0x02;
	IOWIN_REG &= 0xfffeffff; // 16. bit auf 0 setzen -> aktiv
}

bool IOAPIC::status(IOAPIC::Slot interrupt) {
	IOREGSEL_REG = 0x10 + interrupt*0x02;
	return (!(IOWIN_REG & 0x00010000)); // richtig?
}
