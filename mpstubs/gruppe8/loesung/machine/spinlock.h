// $Date: 2009-11-10 09:52:05 +0100 (Tue, 10 Nov 2009) $, $Revision: 2314 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Spinlock
 */

#ifndef __spinlock_include__
#define __spinlock_include__

/*! \brief Mit Hilfe eines Spinlocks kann man Codeabschnitte serialisieren die 
 *  echt nebenläufig auf mehreren CPUs laufen. 
 *  
 *  Die Synchronisation läuft dabei über eine Sperrvariable. Sobald jemand den
 *  kritischen Abschnitt betreten will, setzt er die Sperrvariable. Verlässt 
 *  er den kritischen Abschnitt, so setzt er sie wieder zurück. Ist die 
 *  Sperrvariable jedoch schon gesetzt, dann wartet er aktiv darauf, dass sie 
 *  der Besitzer des kritischen Abschnittes beim Verlassen wieder zurücksetzt.
 *  
 *  Zur Implementierung können die beiden GCC-Intrinsics 
 *  <b> __sync_lock_test_and_set(unsigned int* lock_status, unsigned int value)</b>
 *  und <b> __sync_lock_release(unsigned int* lock_status)</b> verwendet werden.
 *  Diese werden vom Compiler in die architekturspezifischen atomaren 
 *  Operationen übersetzt.
 *  
 *  <a href="http://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Atomic-Builtins.html">Eintrag im GCC Manual über Atomic Builtins</a>
 */
class Spinlock {
private:
    Spinlock(const Spinlock& copy); //verhindert Kopieren
	unsigned int *lock_status;
	unsigned int value;
public:
	~Spinlock() {};
	Spinlock();
	void lock();
	void unlock();
};
 
#endif
