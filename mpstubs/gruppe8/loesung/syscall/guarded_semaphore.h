// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Guarded_Semaphore_include__
#define __Guarded_Semaphore_include__

/*! \file 
 *  \brief Enthaelt die Klasse Guarded_Semaphore
 */

#include "guard/secure.h"
#include "meeting/semaphore.h"

/*! \brief Systemaufrufschnittstelle zum Semaphor
 * 
 *  Die Klasse Guarded_Semaphore implementiert die Systemaufrufschnittstelle zur
 *  Semaphore Klasse. Die von Guarded_Semaphore angebotenen Methoden werden 
 *  direkt auf die Methoden der Basisklasse abgebildet, nur dass ihre Ausfuehrung
 *  jeweils mit Hilfe eines Objekts der Klasse Secure geschuetzt wird. 
 */
class Guarded_Semaphore : public Semaphore
{
private:
    Guarded_Semaphore(const Guarded_Semaphore &copy); // Verhindere Kopieren
public:
	Guarded_Semaphore(int c) : Semaphore(c) {} 
	void p() { wait(); }
	void v() { signal(); }
	void wait();
	void signal();
};

#endif
