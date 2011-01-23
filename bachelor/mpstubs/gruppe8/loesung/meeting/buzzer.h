// $Date: 2009-12-01 18:41:00 +0100 (Tue, 01 Dec 2009) $, $Revision: 2376 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Buzzer_include__
#define __Buzzer_include__

/*! \file
 *  \brief Enthaelt die Klasse Buzzer.
 */

#include "meeting/waitingroom.h"
#include "meeting/bell.h"
#include "meeting/bellringer.h"
#include "syscall/guarded_organizer.h"

/*! \brief Synchronisationsobjekt zum Schlafenlegen fuer eine bestimmte 
 *  Zeitspanne
 *  
 *  Ein "Wecker" ist ein Synchronisationsobjekt, mit dem ein oder mehrere 
 *  Threads sich fuer eine bestimmte Zeit schlafen legen koennen.
 */ 
extern Bellringer bellringer;
extern Guarded_Organizer organizer;

class Buzzer : public Waitingroom, public Bell
{
private:
	Buzzer(const Buzzer &copy); // Verhindere Kopieren

public:
	inline Buzzer() {}
	virtual ~Buzzer();
	void ring();
	void set(int ms);
	void sleep();
	void sleep(int ms);
};

#endif
