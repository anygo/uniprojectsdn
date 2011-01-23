// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 * 
 *  \brief Enthaelt die Klasse Entrant
 */

#ifndef __entrant_include__
#define __entrant_include__

/* Hier muesst ihr selbst Code vervollstaendigen */
#include "thread/coroutine.h"
#include "object/chain.h"

/*! \brief Eine Koroutine, die vom Scheduler verwaltet wird. 
 * 
 *  Die Klasse Entrant erweitert die Klasse Coroutine um die Moeglichkeit, in 
 *  einfach verkettete Listen eingetragen zu werden, insbesondere auch in die
 *  Ready-Liste des Schedulers. Die Verkettungsmoeglichkeit wird durch die 
 *  Ableitung von Chain erreicht.
 */
class Entrant : public Coroutine,  public Chain {

private:
/* Hier muesst ihr selbst Code vervollstaendigen */    
public:
	Entrant(void *tos) : Coroutine(tos) {}
};
#endif
