// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "device/cgastr.h"

CGA_Stream::CGA_Stream(int from_col, int to_col, int from_row, int to_row, bool use_cursor) : CGA_Screen(from_col, to_col, from_row, to_row, use_cursor), O_Stream() {
	attr = STD_ATTR;
	clearScreen();
}


CGA_Stream::~CGA_Stream() {
}

void CGA_Stream::flush() {
	print(buffer, pos, attr);
	pos = 0;
}

void CGA_Stream::setColor(CGA_Screen::color bg, CGA_Screen::color fg, bool blink) {
	attr = bg << 4 | fg;
	attr &= 0x7f;
}

void CGA_Stream::setColor(unsigned char newattr) {
	attr = newattr;
}

unsigned char CGA_Stream::getColor() {
	return attr;
}
