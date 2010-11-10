// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Enthaelt die Klasse Bellringer.
 */

#ifndef __Bellringer_include__
#define __Bellringer_include__

#include "object/list.h"
#include "meeting/bell.h"

/*! \brief Verwaltung und Anstossen von zeitgesteuerten Aktivitaeten.
 * 
 *  Der "Gloeckner" (Bellringer) wird regelmaessig aktiviert und prueft, ob 
 *  irgendwelche "Glocken" (Bell-Objekte) laeuten muessen. Die Glocken befinden 
 *  sich in einer Liste, die der Gloeckner verwaltet. 
 */

class Bellringer : public List
{
private:
    Bellringer(const Bellringer &copy); // Verhindere Kopieren
public:
    /*! \brief Konstruktor.
     */
    Bellringer() {}
	void check();
	void job(Bell *bell, int ticks);
	void cancel(Bell *bell);
	void ring_the_bells();
};

#endif
