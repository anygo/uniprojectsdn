// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Application 
 */

#ifndef __application_include__
#define __application_include__

#include "thread/coroutine.h"
#include "syscall/thread.h"

/*! \brief Die Klasse Application definiert die einzige Anwendung von OO-Stubs.
 */
class Application : public Thread { 
 
private:
    Application (const Application &copy); // Verhindere Kopieren
	int ID;
public:
	Application(int id, void *tos) : Thread(tos) {
		ID = id;
	}
    void action();
};

#endif
