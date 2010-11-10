// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file 
 *  \brief Enthält Debugmacros, um Debugausgaben auf einen eigenen
 *  Screen umzuleiten.
 *  
 *  Für den Uniprozessorfall reicht es ein CGA_Stream Objekt für Debugausgaben
 *  (\b dout) anzulegen. Für den Multiprozessorfall soll jedoch für jede CPU ein
 *  Objekt für Debugausgaben angelegt werden. Das Debugmacro muss dann mit Hilfe
 *  von APICSystem::getCPUID() die Ausgabe auf das entsprechende Objekt 
 *  umleiten. Dazu kann der <b>? : </b>Operator verwendet werden.
 */

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include "device/cgastr.h"
#include "machine/apicsystem.h"

extern CGA_Stream dout_CPU0;
extern CGA_Stream dout_CPU1;
extern CGA_Stream dout_CPU2;
extern CGA_Stream dout_CPU3;
extern APICSystem system;

extern char drehding[];

#define DEBUG

#ifdef DEBUG
	#define DBG (  (((int)system.getCPUID()) == 0) ? dout_CPU0 : (((int)system.getCPUID()) == 1) ? dout_CPU1 : (((int)system.getCPUID()) == 2) ? dout_CPU2 : dout_CPU3 )
#else
	#define DBG if (false) dout_CPU0
#endif
 
#endif // __DEBUG_H__
