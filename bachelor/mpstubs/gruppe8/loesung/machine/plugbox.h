// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Plugbox
 */

#ifndef __Plugbox_include__
#define __Plugbox_include__

#include "guard/gate.h"

/*! \brief Abstraktion einer Interruptvektortabelle. 
 * 
 *  Damit kann man die Adresse der Behandlungsroutine fuer jeden Hardware- 
 *  und Softwareinterrupt und jede Prozessorexception festlegen. Jede 
 *  Unterbrechungsquelle wird durch ein Gate-Objekt repräsentiert. Diese 
 *  liegen in einem Feld(128 Elemente). Der Index in diesen Feld ist dabei
 *  die Vektornummer. 
 */
class Plugbox {
private:
    Plugbox(const Plugbox &copy); // Verhindere Kopieren
	Gate *gate_map[128];

public:
	enum Vector { timer = 32, keyboard = 33, assassin = 100, wakeup = 101 };
	Plugbox();
	void assign(unsigned int slot, Gate &gate);
	Gate & report(unsigned int slot);
};

#endif
