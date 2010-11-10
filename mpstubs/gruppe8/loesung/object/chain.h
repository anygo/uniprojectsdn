// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Chain
 */
#ifndef __chain_include__
#define __chain_include__

/*! \brief Verkettungszeiger zum Einfügen eines Objektes in eine einfach
 *  verkettete Liste.
 *  
 *  Die Klasse Chain stellt einen Verkettungszeiger auf ein weiteres 
 *  Chain Element zur Verfügung und ist damit Basis aller Klassen, deren
 *  Instanzen in Listen (Queue Objekten) verwaltet werden sollen. 
 */ 
class Chain {
private:
//    Chain(const Chain &copy); // Verhindere Kopieren
public:
    Chain() {}
public:
    /*! \brief next gibt das nächste Chain Element der Liste an. 
     * 
     *  Wenn kein nächstes Element existiert, sollte next ein Nullzeiger sein.
     */
    Chain* next;
};

#endif

