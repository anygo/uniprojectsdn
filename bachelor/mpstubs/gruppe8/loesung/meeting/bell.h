// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __Bell_include__
#define __Bell_include__

/*! \file
 *  \brief Enthaelt die Klasse Bell.
 */

#include "object/chain.h"

/*! \brief Ermoeglicht ein zeitgesteuertes Ausloesen einer Aktivitaet.
 * 
 *  Eine "Glocke" ist eine abstrakte Basisklasse, die das zeitgesteuerte 
 *  Ausloesen einer Aktivitaet erlaubt. Dazu besitzt sie intern einen Zaehler, der
 *  vom "Gloeckner" (Bellringer) verwaltet wird. 
 *  
 *  \note 
 *  Um Bell verwenden zu koennen, muss eine abgeleitete Klasse erstellt werden in
 *  der die Methode ring() definiert wird.
 *  Alle anderen Methoden werden am besten inline definiert. 
 */

class Bell : public Chain
/* Hier muesst ihr selbst Code vervollstaendigen */ 	//WAS DENN
{
private:
    Bell(const Bell &copy); // Verhindere Kopieren
public:
    /*! \brief Konstruktor
     */
	int counter; 
    Bell() {}
	inline void wait(int value) { counter = value; }
	inline int wait() { return counter; }
	void tick() { counter--; }
	inline bool run_down() { return (counter == 0); }
	virtual void ring()=0;

};

#endif
