// $Date: 2009-11-09 09:39:49 +0100 (Mon, 09 Nov 2009) $, $Revision: 2310 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse IOAPIC zum Zugriff auf den IO-APIC
 */

#ifndef __IOAPIC_H__
#define __IOAPIC_H__

#include "machine/ioapic_registers.h"
#include "machine/plugbox.h"
#include "machine/apicsystem.h"

/*! \brief Abstraktion des IO-APICs, der zur Verwaltung der externen Interrupts
 *  dient. 
 *  
 *  Kernstück des IOAPICs ist die IO-Redirection Table. Dort lässt sich frei 
 *  konfigurieren, welchem Interruptvektor eine bestimmte externe 
 *  Unterbrechung zugeordnet werden soll. Ein Eintrag in dieser Tabelle ist
 *  64 Bit breit. struct IOREDTBL_L und struct IOREDTBL_H sind Bitfelder, die
 *  die einzelnen Einstellungen eines Eintrages zugänglich machen. 
 */

class IOAPIC {
private:
    IOAPIC(const IOAPIC& copy); //Verhindere Kopieren

public:
	IOAPIC() {} //tut nichts
	enum Slot { keyboard = 1, timer = 2};
	void init();
	void config(IOAPIC::Slot slot, Plugbox::Vector vector);
	void allow(IOAPIC::Slot interrupt);
	void forbid(IOAPIC::Slot interrupt);
	bool status(IOAPIC::Slot interrupt);

};

#endif
