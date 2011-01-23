// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Diese Datei enthält die Klasse Guard.
 */

#ifndef __Guard_include__
#define __Guard_include__

#include "machine/spinlock.h"
#include "object/queue.h"
#include "guard/gate.h"
#include "guard/locker.h"

/*! \brief Synchronisation des BS-Kerns mit Unterbrechungen. 
 * 
 *  Die Klasse Guard dient der Synchronisation zwischen "normalen"
 *  Kernaktivitäten (zur Zeit Ausgaben, später Systemaufrufe) und 
 *  Unterbrechungsbehandlungsroutinen. Dazu besitzt Guard eine Warteschlange 
 *  (ein Queue Objekt), in die Gate Objekte eingereiht werden können. Das ist 
 *  immer dann erforderlich, wenn zum Zeitpunkt des Auftretens einer 
 *  Unterbrechung der kritische Abschnitt gerade besetzt ist, die epilogue()
 *  Methode also nicht sofort bearbeitet werden darf. Die angesammelten 
 *  Epiloge werden behandelt, sobald der kritische Abschnitt wieder 
 *  freigegeben wird. 
 *  
 *  \par Hinweise
 *  - Die Epilogqueue stellt eine zentrale Datenstruktur dar, deren Konsistenz
 *    geeignet gesichert werden muß. Die von uns durch die Klasse Queue 
 *    bereitgestellte Implementierung ist nicht unterbrechungstransparent! 
 *    Entweder ihr implementiert also selbst eine unterbrechungstransparente
 *    Queue, oder ihr synchronisiert die bereitgestellte Queue entsprechend hart.
 *    Im Multiprozessorfall muss zusätzlich noch auf die Synchronisation mit 
 *    anderen CPUs geachtet werden.  
 *  - Da Gate Objekte nur einen einzigen Verkettungszeiger besitzen, dürfen sie 
 *    zu einem Zeitpunkt nur ein einziges Mal in der Epilogliste aufgeführt
 *    sein. Wenn also zwei gleichartige Interrupts so schnell aufeinanderfolgen,
 *    dass der zugehörige Epilog noch gar nicht behandelt wurde, darf 
 *    nicht versucht werden, dasselbe Gate Objekt zweimal in die Epilogliste
 *    einzutragen. Die Klasse Gate  bietet Methoden, dies zu vermerken bzw.
 *    zu prüfen.
 *  - Ein Betriebssystem sollte Unterbrechungen immer nur so kurz wie möglich
 *    sperren. Daher sieht das Pro-/Epilog-Modell vor, dass Epiloge durch 
 *    Prologe unterbrochen werden können. Für OOStuBS bedeutet das, dass bereits
 *    vor der Ausführung des Epilogs einer Unterbrechungsbehandlung Interrupts 
 *    wieder zugelassen werden sollten. 
 */
class Guard : public Locker {
private:
    Guard (const Guard &copy); // Verhindere Kopieren
	Spinlock guard_lock;
	Queue epilogue_queue;
	bool leave2();
public:
    /*! \brief Konstruktor
     */
    Guard() {}
	~Guard() {}

	void enter();
	void leave();
	void relay (Gate *item);
	void try_idling();
};

#endif
