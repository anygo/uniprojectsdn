// $Date: 2009-09-25 16:32:11 +0200 (Fri, 25 Sep 2009) $, $Revision: 2224 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __assassin_h__
#define __assassin_h__

#include "guard/gate.h"

class Assassin : public Gate {
public:
	void hire();
	bool prologue();
	virtual void epilogue();
};
#endif
