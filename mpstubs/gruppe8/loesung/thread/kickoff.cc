// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "guard/guard.h"
#include "object/debug.h"
#include "thread/coroutine.h"

extern Guard guard;

void kickoff(Coroutine *obj) {
	guard.leave();
	obj->action();
}
