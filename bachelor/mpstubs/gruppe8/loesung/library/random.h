// $Date: 2009-10-09 09:21:41 +0200 (Fri, 09 Oct 2009) $, $Revision: 2233 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __random_include__
#define __random_include__

/*! \file 
 *  \brief Enthaelt die Klasse Random
 */

class Random {
private:
    Random(const Random &copy); // Verhindere Kopieren
private:
    unsigned long r0, r1, r2, r3, r4, r5, r6;
    unsigned long multiplier, addend, ic_state;
public:
    /*! \brief Konstruktor; Initialisierung mit \b seed
     *  \param seed Initialwert fuer den Pseudozufallszahlengenerator.
     */
    Random(int seed);
    
    /*! \brief Liefert eine Zufallszahl.
     *  \return Zufallszahl.
     */
    int number();
};

#endif
