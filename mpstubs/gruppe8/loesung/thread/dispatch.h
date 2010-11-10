// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __dispatch_include__
#define __dispatch_include__

/*! \file
 * 
 *  \brief Enthaelt die Klasse Dispatcher.
 */

#include "thread/coroutine.h"
#include "machine/apicsystem.h"

/*! \brief Implementierung des Dispatchers
 *  
 *  Der Dispatcher verwaltet den Life-Pointer, der die jeweils aktive Koroutine 
 *  angibt und fuehrt die eigentlichen Prozesswechsel durch. In der 
 *  Uniprozessorvariante wird nur ein einziger Life-Pointer benoetigt, da 
 *  lediglich ein Prozess auf einmal aktiv sein kann. Fuer die 
 *  Mehrprozessorvariante wird hingegen fuer jede CPU ein eigener Life-Pointer
 *  benoetigt. 
 *  
 */
class Dispatcher {
private:
    Dispatcher(const Dispatcher &copy); // Verhindere Kopieren

public:
 	Coroutine *life[CPU_MAX]; 
	
	inline Dispatcher() {
		for (int i = 0; i < CPU_MAX; i++) {
			life[i] = 0;
		}
	}

	Coroutine * active();
	void go(Coroutine &first);
	void dispatch(Coroutine &next);
};

#endif
