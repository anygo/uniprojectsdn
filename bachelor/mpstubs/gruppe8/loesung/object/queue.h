// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#ifndef __queue_include__
#define __queue_include__

#include "object/chain.h"

/*! \brief Die Klasse Queue realisiert eine einfach verkettete Liste von 
 *  (sinnvollerweise spezialisierten) Chain Objekten.
 *  
 *  Die Implementierung ist etwas trickreich (siehe Vorlesung). tail verweist
 *  nämlich nicht, wie oft üblich, auf das letzte Element der Liste, sondern 
 *  auf den next Zeiger des letzten Elements, bzw., solange die Liste noch leer
 *  ist, auf den head Zeiger der Liste. Dadurch muss beim Einfügen eines 
 *  Elements an die Liste nicht überprüft werden, ob bereits Elemente in ihr 
 *  enthalten sind. Beim Entfernen von Elementen kann auf die Fallunterscheidung
 *  allerdings nicht verzichtet werden.
 */
class Queue {
private:
    Queue(const Queue &copy); // Verhindere Kopieren
protected:
    Chain* head;
    Chain** tail;
public:
    /*! \brief Der Konstruktor initialisiert die Liste als leere Liste
     */
    Queue () { head = 0; tail = &head; }
    /*! \brief Das Listenelement item wird an das Ende der Liste angefügt.
     *  \param item Listenelement, welches angefügt werden soll.
     */
    void enqueue (Chain* item);
    /*! \brief Liefert das erste Element der Liste und entfernt es gleichzeitig
     *  aus dieser.
     *  \return entferntes Kopfelement.
     */
    Chain* dequeue ();
    /*! \brief Mit dieser Methode kann das angegebene Element item  aus der 
     *  Liste entfernt werden, unabhängig von seiner Position dort.
     *  \param item Element, welches entfernt werden soll.
     *  \return Gibt an, ob etwas entfernt wurde. 
     */
    bool remove (Chain* item);
};

#endif

