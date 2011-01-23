// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Enthaelt die Klasse Thread
 */
#ifndef __thread_include__
#define __thread_include__

#include "thread/customer.h"

class Thread : public Customer
{
private:
    Thread(const Thread &copy); // Verhindere Kopieren

public:
	Thread(void *tos) : Customer(tos) {}
};

#endif
