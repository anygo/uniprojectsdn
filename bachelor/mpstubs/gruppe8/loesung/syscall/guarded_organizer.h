// $Date: 2009-11-27 16:35:50 +0100 (Fri, 27 Nov 2009) $, $Revision: 2365 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Guarded_Organizer_include__
#define __Guarded_Organizer_include__

/*! \file
 *  \brief Enthaelt die Klasse Guarded_Organizer
 */
#include "thread/organizer.h"
#include "guard/secure.h"

/*! \brief Systemaufrufschnittstelle zum Organizer
 * 
 *  Der Guarded_Organizer implementiert die Systemaufrufschnittstelle zum 
 *  Organizer. Die von ihm angebotenen Methoden werden direkt auf die Methoden 
 *  der Basisklasse abgebildet, nur dass ihre Ausfuehrung jeweils mit Hilfe eines
 *  Objekts der Klasse Secure geschuetzt wird und dass nicht Customer, sondern 
 *  Thread Objekte behandelt werden.
 *  
 *  \note 
 *  Die Klasse Guarded_Organizer ersetzt die Klasse Guarded_Scheduler aus
 *  Aufgabe 5. 
 */
class Guarded_Organizer : public Organizer
{
private:
    Guarded_Organizer(const Guarded_Organizer &copy); // Verhindere Kopieren
public:
    Guarded_Organizer() {}
	void exit();
	void kill(Thread &thread);
	void resume();
	void sleep_until_IRQ();
};

#endif
