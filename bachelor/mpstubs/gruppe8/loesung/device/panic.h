// $Date: 2009-08-17 12:16:56 +0200 (Mon, 17 Aug 2009) $, $Revision: 2210 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Panic
 */
#ifndef __panic_include__
#define __panic_include__

/* INCLUDES */

#include "guard/gate.h"

/*! \brief Standardunterbrechungsbehandlung
 * 
 *  Die Klasse Panic dient der Behandlung von Unterbrechungen und Ausnahmen. 
 *  Nach der Ausgabe einer Fehlermeldung wird der Prozessor angehalten. Bei der
 *  Initialisierung der Plugbox wird diese Form der Unterbrechungsbehandlung 
 *  für alle Interrupt Nummern eingetragen. 
 */
class Panic : public Gate {
private:
    Panic(const Panic &copy); // Verhindere Kopieren
public:
	void trigger(); 
	Panic() {}

	bool prologue(); 
	void epilogue() {}
 };

#endif

