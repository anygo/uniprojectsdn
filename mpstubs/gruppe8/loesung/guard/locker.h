// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Diese Datei enthält die Klasse Locker.
 */

#ifndef __Locker_include__
#define __Locker_include__

/*! \brief Die Klasse Locker dient dem Schutz kritischer Abschnitte. 
 * 
 *  Dazu verwaltet sie eine Sperrvariable, die angibt, ob der zu schützende 
 *  kritische Abschnitt gerade frei oder besetzt ist.
 *   
 *  Die Klasse Locker bestimmt jedoch nicht, was zu tun ist, wenn der kritische
 *  Abschnitt besetzt ist. Ebenso trifft sie keine Vorkehrungen, um ihre 
 *  eigenen kritischen Abschnitte zu schützen.
 *  
 *  \par Hinweise
 *  Die Methoden der Klasse sind so kurz, dass sie am besten inline 
 *  definiert werden sollten. 
 */
class Locker {
private:
    Locker(const Locker &copy); // Verhindere Kopieren

protected:
	bool lock;

public:
	Locker() {  lock = false; }
	inline void enter() { lock = true; }
	inline void retne() { lock = false; }
	inline bool avail() const { return !(lock); }
};

#endif
