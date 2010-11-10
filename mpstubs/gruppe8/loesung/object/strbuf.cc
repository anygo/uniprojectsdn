// $Date: 2009-08-11 16:57:46 +0200 (Tue, 11 Aug 2009) $, $Revision: 2208 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "object/strbuf.h"

/* Hier muesst ihr selbst Code vervollstaendigen*/

Stringbuffer::Stringbuffer() {
	pos = 0;
}

Stringbuffer::~Stringbuffer() {
}

void Stringbuffer::put(char c) {
	buffer[pos++] = c;
	if (pos == 79) {
		flush();
	}
}

