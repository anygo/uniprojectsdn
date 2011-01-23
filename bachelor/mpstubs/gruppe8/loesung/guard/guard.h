// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Diese Datei enth�lt die Klasse Guard.
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
 *  Kernaktivit�ten (zur Zeit Ausgaben, sp�ter Systemaufrufe) und 
 *  Unterbrechungsbehandlungsroutinen. Dazu besitzt Guard eine Warteschlange 
 *  (ein Queue Objekt), in die Gate Objekte eingereiht werden k�nnen. Das ist 
 *  immer dann erforderlich, wenn zum Zeitpunkt des Auftretens einer 
 *  Unterbrechung der kritische Abschnitt gerade besetzt ist, die epilogue()
 *  Methode also nicht sofort bearbeitet werden darf. Die angesammelten 
 *  Epiloge werden behandelt, sobald der kritische Abschnitt wieder 
 *  freigegeben wird. 
 *  
 *  \par Hinweise
 *  - Die Epilogqueue stellt eine zentrale Datenstruktur dar, deren Konsistenz
 *    geeignet gesichert werden mu�. Die von uns durch die Klasse Queue 
 *    bereitgestellte Implementierung ist nicht unterbrechungstransparent! 
 *    Entweder ihr implementiert also selbst eine unterbrechungstransparente
 *    Queue, oder ihr synchronisiert die bereitgestellte Queue entsprechend hart.
 *    Im Multiprozessorfall muss zus�tzlich noch auf die Synchronisation mit 
 *    anderen CPUs geachtet werden.  
 *  - Da Gate Objekte nur einen einzigen Verkettungszeiger besitzen, d�rfen sie 
 *    zu einem Zeitpunkt nur ein einziges Mal in der Epilogliste aufgef�hrt
 *    sein. Wenn also zwei gleichartige Interrupts so schnell aufeinanderfolgen,
 *    dass der zugeh�rige Epilog noch gar nicht behandelt wurde, darf 
 *    nicht versucht werden, dasselbe Gate Objekt zweimal in die Epilogliste
 *    einzutragen. Die Klasse Gate  bietet Methoden, dies zu vermerken bzw.
 *    zu pr�fen.
 *  - Ein Betriebssystem sollte Unterbrechungen immer nur so kurz wie m�glich
 *    sperren. Daher sieht das Pro-/Epilog-Modell vor, dass Epiloge durch 
 *    Prologe unterbrochen werden k�nnen. F�r OOStuBS bedeutet das, dass bereits
 *    vor der Ausf�hrung des Epilogs einer Unterbrechungsbehandlung Interrupts 
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
