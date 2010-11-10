// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Diese Datei enthält die Klasse Keyboard
 */

#ifndef __Keyboard_include__
#define __Keyboard_include__

#include "machine/keyctrl.h"
#include "guard/gate.h"
#include "machine/key.h"
#include "syscall/guarded_semaphore.h"
 
/*! \brief Die Klasse Keyboard stellt die Abstraktion der Tastatur dar.
 * 
 *  Sie sorgt für die korrekte Initialisierung und vor allem für die 
 *  Unterbrechungsbehandlung. Später wird Keyboard auch die Tastaturabfrage
 *  durch die Anwendung ermöglichen.
 */
class Keyboard : public Gate, public Keyboard_Controller {        
private:
	Keyboard (const Keyboard &copy); // Verhindere Kopieren
	Key currentkey;
	Guarded_Semaphore sema;		// wie aufrufen?
	Key puf;

public:
	Keyboard() : sema(0) {}
	virtual ~Keyboard() {}
 	void plugin();

	bool prologue();
	void epilogue();
	Key getkey();
 };

#endif
