// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Guarded_Buzzer_include__
#define __Guarded_Buzzer_include__

/*! \file
 *  \brief Enthaelt die Klasse Guarded_Buzzer
 */

#include "meeting/buzzer.h"

/*! \brief Schnittstelle von Anwendungsthreads zu Buzzer-Objekten.
 * 
 *  Die Klasse Guarded_Buzzer implementiert die Systemaufrufschnittstelle zur 
 *  Buzzer Klasse. Die von Guarded_Buzzer angebotenen Methoden werden direkt auf
 *  die Methoden der Basisklasse abgebildet, nur dass ihre Ausfuehrung jeweils 
 *  mit Hilfe eines Objekts der Klasse Secure geschuetzt wird. 
 */
class Guarded_Buzzer : public Buzzer
{
private:
    Guarded_Buzzer(const Guarded_Buzzer &copy); // Verhindere Kopieren
public:
    Guarded_Buzzer() {}
	~Guarded_Buzzer();
	void set(int ms);
	void sleep();
};

#endif
