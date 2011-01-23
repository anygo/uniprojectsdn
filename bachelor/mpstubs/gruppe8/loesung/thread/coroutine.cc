// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#include "thread/coroutine.h"
#include "device/cgastr.h"

extern CGA_Stream kout;

/* Hier muesst ihr selbst Code vervollstaendigen */ 

Coroutine::Coroutine(void *tos) {
	toc_settle(&(this->regs), tos, kickoff, this);
	is_dying = false;
}

void Coroutine::go() {
	toc_go(&(this->regs));
}

void Coroutine::resume(Coroutine &next) {
	toc_switch(&(this->regs), &(next.regs));
}

void Coroutine::set_kill_flag() {
	is_dying = true;
}

void Coroutine::reset_kill_flag() {
	is_dying = false;
}

bool Coroutine::dying() {
	return is_dying;
}
