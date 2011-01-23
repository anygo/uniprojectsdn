// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __waitingroom_include__
#define __waitingroom_include__

/*! \file
 *  \brief Enthaelt die Klasse Waitingroom.
 */

/* Hier muesst ihr selbst Code vervollstaendigen */ 
#include "object/queue.h"


/*! \brief Liste von Threads, die auf ein Ereignis warten. 
 *  
 *  Die Klasse Waitingroom implementiert eine Liste von Prozessen (Customer 
 *  Objekten), die alle auf ein bestimmtes Ereignis warten.
 *  \note
 *  Die Methode remove(Customer*) muss virtuell sein, damit der Organizer einen
 *  Prozess aus dem Wartezimmer entfernen kann, ohne wissen zu muessen, welcher 
 *  Art dieses Wartezimmer ist. Sofern es erforderlich ist, kann eine von 
 *  Waitingroom abgeleitete Klasse die Methode auch noch neu definieren.
 * 
 *  Der Destruktor sollte wie bei allen Klassen, die virtuelle Methoden 
 *  definieren, ebenfalls virtuell sein. 
 */
class Customer;
class Waitingroom : public Queue
{
private:
    Waitingroom(const Waitingroom &copy); // Verhindere Kopieren
public:
    Waitingroom() {}
	virtual ~Waitingroom();
	virtual void remove(Customer *customer);
};

#endif
