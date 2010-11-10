// $Date: 2009-09-15 18:43:03 +0200 (Tue, 15 Sep 2009) $, $Revision: 2219 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Gate
 */

#ifndef __Gate_include__
#define __Gate_include__


#include "object/chain.h"

class Gate : public Chain {
private:
	bool inQueue;

public:
	Gate() {}
	virtual ~Gate() {};
//	virtual void trigger()=0;

	// Aufgabe 3:
	virtual bool prologue()=0;
	inline virtual void epilogue() {}
	inline void queued(bool q) {inQueue = q; }
	inline bool queued() { return inQueue; }
};

#endif
