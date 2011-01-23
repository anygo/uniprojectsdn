// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Diese Datei enthält die Klasse Secure.
 */

#ifndef __Secure_include__
#define __Secure_include__

#include "guard/guard.h"

extern Guard guard;

/*! \brief Die Klasse Secure dient dem bequemen Schutz kritischer Abschnitte.
 * 
 *  Dabei wird die Tatsache ausgenutzt, dass der C++ Compiler für jedes Objekt
 *  automatisch Konstruktor- und Destruktoraufrufe in den Code einbaut und dass
 *  ein Objekt seine Gültigkeit verliert, sobald der Bereich (Scope), in dem es
 *  deklariert wurde, verlassen wird.
 * 
 *  Wenn im Konstruktor von Secure also ein kritischer Abschnitt betreten und 
 *  im Destruktor wieder verlassen wird, kann die Markierung kritischer 
 *  Codebereiche ganz einfach folgendermaßen erfolgen:
 * 
 *  \verbatim
    // unkritisch
    ...
    { Secure section;
       // hier kommen die kritischen Anweisungen 
       ...
    }
    // Ende des kritischen Abschnitts
    \endverbatim 
 *  \par Hinweis
 *  Die Methoden der Klasse sind so kurz, dass sie am besten inline definiert 
 *  werden sollten. 
 */
class Secure {
private:
    Secure(const Secure &copy); // Verhindere Kopieren
	
public:
	inline Secure() { guard.enter(); }
	inline ~Secure() { guard.leave(); }
};

#endif
