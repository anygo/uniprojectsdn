// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __List_include__
#define __List_include__

/*! \file
 *  \brief Enthaelt die Klasse List
 */

#include "object/queue.h"

/*! \brief Implementierung einer einfach verketteten Liste. 
 * 
 *  Die Klasse List realisiert eine einfach verkettete Liste von 
 *  (sinnvollerweise spezialisierten) Chain Objekten. Im Gegensatz zu Queue 
 *  koennen Elemente jedoch auch am Anfang oder in der Mitte eingefuegt werden. 
 */
class List : public Queue {
private:
    List(const List &copy); // Verhindere Kopieren
public:
    /*! \brief Konstruktor.
     *  Der Konstruktor initialisiert die Liste als leere Liste.
     */
    List() {}
public:
    /*! \brief Liefert das erste Element der Liste ohne es zu entfernen.
     *  \return Erstes Element der Liste.
     */
    Chain* first() { return head; }

    /*! \brief Fuegt das \b new_item am Anfang der Liste ein.
     *  \param new_item Einzufuegendes Element.
     */
    void insert_first(Chain* new_item);

    /*! \brief Fuegt das Element new_item hinter dem Element old_item in die 
     *  Liste ein.
     *  \param old_item Element, nach dem eingefuegt werden soll.
     *  \param new_item Einzufuegendes Element.
     */
    void insert_after(Chain* old_item, Chain* new_item);
};

#endif

