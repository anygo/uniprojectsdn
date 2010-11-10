// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Semaphore_include__
#define __Semaphore_include__

/*! \file
 *  \brief Enthaelt die Klasse Semaphore.
 */

#include "meeting/waitingroom.h"
#include "thread/customer.h"

/*! \brief Semaphore werden zur Synchronisation von Threads verwendet.
 *  
 *  Die Klasse Semaphore implementiert das Synchronisationskonzept des zaehlenden
 *  Semaphors. Die benoetigte Warteliste erbt sie dabei von ihrer Basisklasse 
 *  Waitingroom. 
 */
class Semaphore : public Waitingroom
{
private:
    Semaphore(const Semaphore &copy); // Verhindere Kopieren
	int counter;
public:
	Semaphore(int c) : counter(c) {}
	void p() { wait(); }
	void v() { signal(); }
	void wait();
	void signal();
	void remove(Customer *customer);
};

#endif
