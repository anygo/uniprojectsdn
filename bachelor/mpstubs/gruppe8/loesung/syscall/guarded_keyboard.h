// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Guarded_Keyboard_include__
#define __Guarded_Keyboard_include__

/*! \file
 *  \brief Enthaelt die Klasse Guarded_Keyboard
 */
#include "device/keyboard.h"

/*! \brief Systemaufrufschnittstelle zur Tastatur
 */
class Guarded_Keyboard : public Keyboard
{
private:
    Guarded_Keyboard (const Guarded_Keyboard &copy); // Verhindere Kopieren
public:
    Guarded_Keyboard() {}
	Key getkey();
};

#endif
