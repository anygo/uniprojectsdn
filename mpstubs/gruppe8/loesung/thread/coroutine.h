// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 * 
 *  \brief Enthaelt die Klasse Coroutine und die Funktion kickoff
 */

#ifndef __Coroutine_include__
#define __Coroutine_include__

#include "machine/toc.h"

/*! \brief Die Klasse Coroutine stellt die Abstraktion einer Koroutine dar. 
 * 
 *  Sie ermoeglicht die Prozessorabgabe an eine andere Koroutine und stellt 
 *  durch die Struktur toc Speicherplatz zur Verfuegung, um die Inhalte der 
 *  nicht-fluechtigen Register bis zur naechsten Aktivierung zu sichern. Ausserdem
 *  sorgt sie fuer die Initialisierung dieser Registerwerte, damit bei der ersten
 *  Aktivierung die Ausfuehrung an der richtigen Stelle und mit dem richtigen
 *  Stack beginnt.  
 */ 
class Coroutine {
	struct toc regs;

private:
	bool is_dying;

public:
	Coroutine(void *tos);
	virtual void action()=0;
	void go();
	void resume(Coroutine &next);
	void set_kill_flag();
	void reset_kill_flag();
	bool dying();
};

void kickoff(Coroutine *obj); 

#endif
