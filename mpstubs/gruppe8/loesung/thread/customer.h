// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __customer_include__
#define __customer_include__

/*! \file
 *  \brief Enthaelt die Klasse Customer
 */
#include "meeting/waitingroom.h"
#include "thread/entrant.h"

/*! \brief Ein Thread, der auf ein Ereignis warten kann.
 * 
 *  Die Klasse Customer erweitert die Klasse Entrant um die Moeglichkeit, ein 
 *  Ereignis, auf das der betreffende Prozess wartet, zu vermerken und 
 *  abzufragen. 
 */
class Customer : public Entrant
{
private:
	Waitingroom *room;

public:
	inline Customer(void *tos) : Entrant(tos) { room = 0; }	
	inline Waitingroom *waiting_in() { return room; } 
	inline void waiting_in(Waitingroom *w) { room = w; }
};

#endif
