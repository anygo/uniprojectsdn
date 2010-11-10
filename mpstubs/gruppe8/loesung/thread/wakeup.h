// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __WAKEUP_H__
#define __WAKEUP_H__

/*! \file
 *  \brief Enthaelt die Klasse WakeUp
 */
#include "machine/plugbox.h"
#include "guard/gate.h"

/*! \brief Interruptbehandlungsobjekt, um in MPStuBS schlafende Prozessoren 
 *  mit einem IPI zu wecken, falls neue Prozesse aktiv wurden.
 *  
 *  Nur in MPStuBS benoetigt.   
 */
extern Plugbox plugbox;

class WakeUp : public Gate
{
	public:
		void activate() { plugbox.assign(Plugbox::wakeup, *this); }
		bool prologue() { return true; }
		void epilogue() { }
};

#endif /* __WAKEUP_H__ */
