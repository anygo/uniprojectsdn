// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Enthaelt die Klasse Watch
 */

#ifndef __watch_include__
#define __watch_include__

/* INCLUDES */

#include "guard/gate.h"
#include "machine/pit.h"

/*! \brief Interruptbehandlung fuer Timerinterrupts.
 *  
 *  Die Klasse Watch sorgt fuer die Behandlung der Zeitgeberunterbrechungen, 
 *  indem sie eine Zeitscheibe verwaltet und gegebenenfalls einen Prozesswechsel
 *  ausloest. 
 */
class Watch : public Gate, public PIT
{
private:
    Watch (const Watch &copy); // Verhindere Kopieren

public:
	inline Watch(int us) : PIT(us) {}
	void windup();
	bool prologue();
	void epilogue();
};

#endif
