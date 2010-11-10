
/*! \file
 *  \brief Enthält die Klasse Santas_House.
 */

#ifndef __SANTAS_HOUSE_H__
#define __SANTAS_HOUSE_H__

#include "syscall/thread.h"
#include "syscall/guarded_vesagraphics.h"

class Santas_House : public Thread {
    void print_house(const Point& p);
public:
    Santas_House(void* tos) : Thread(tos) { }
    virtual void action();
};

#endif
