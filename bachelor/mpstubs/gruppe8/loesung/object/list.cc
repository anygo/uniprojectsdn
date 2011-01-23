// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "object/list.h"

// INSERT_FIRST: Stellt das Element an den Anfang der Liste
void List::insert_first(Chain* new_item) {
    if (head) {             // Die Liste ist nicht leer.
        new_item->next = head;
        head = new_item;
    } else {                // Die Liste ist leer. Dann kann das Element 
        enqueue (new_item); // genausogut hinten angehaengt werden.
    }
}

// INSERT_AFTER: Fuegt das neue Element hinter dem angegebenen alten
//               Element in die Liste ein.
void List::insert_after(Chain* old_item, Chain* new_item) {
    new_item->next = old_item->next;
    old_item->next = new_item;
}
