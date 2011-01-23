// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __organizer_include__
#define __organizer_include__

/*! \file 
 *  \brief Enthaelt die Klasse Organizer.
 */

#include "thread/customer.h"
#include "thread/scheduler.h"


/*! \brief Ein Organizer ist ein spezieller Scheduler, der zusaetzlich das Warten
 *  von Prozessen (Customer) auf Ereignisse erlaubt.
 */ 


class Organizer : public Scheduler
{
private:
    Organizer(const Organizer &copy); // Verhindere Kopieren
public:
    Organizer() {}
	void block(Customer &customer, Waitingroom &waitingroom);
	void wakeup(Customer &customer);
	void kill(Customer &customer);
};

#endif
