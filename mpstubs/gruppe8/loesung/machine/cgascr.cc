// $Date: 2009-08-11 16:57:46 +0200 (Tue, 11 Aug 2009) $, $Revision: 2208 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "machine/cgascr.h"
#include "machine/io_port.h"

/* Hier muesst ihr selbst Code vervollstaendigen */ 

typedef struct cga_char cga_char;
static const char * CGA_START = (const char *)0xb8000;

CGA_Screen::CGA_Screen(int from_col, int to_col, int from_row, int to_row, bool use_cursor) : index_reg(IO_Port(0x3d4)), data_reg(IO_Port(0x3d5)){

	this->from_col = from_col;
	this->to_col = to_col;
	this->from_row = from_row;
	this->to_row = to_row;
	this->use_cursor = use_cursor;
	this->startAdress = (char *)CGA_START + (from_row*80 + from_col)*2;
	this->width = to_col - from_col + 1;
	this->height = to_row - from_row + 1;
	setpos(0, 0);
}

void CGA_Screen::clearScreen(unsigned char attr) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			show(i, j, ' ', attr);
		}
	}
}

void CGA_Screen::setpos(int x, int y) {
	if (!(x >= width)) {
	if (use_cursor) {
		int pos = (from_row + y)*80 + from_col + x;
		index_reg.outb(14);			// in register 14 schreiben
		data_reg.outb(pos >> 8);		// oberste 8 bit in register 14 schrieben
		index_reg.outb(15);
		data_reg.outb(pos);			// untere 8 bit in register 15 schreiben
	}
	}
	software_cursor_x = x;
	software_cursor_y = y;
}

void CGA_Screen::getpos(int &x, int &y) {
	x = software_cursor_x;
	y = software_cursor_y;
}

void CGA_Screen::show(int x, int y, char character, unsigned char attr) {
	char *c = getAdress(x, y);
	*c = character;
	*(c+1) = attr;
}

void CGA_Screen::print(char *string, int length, unsigned char attr) {	
	for (int i = 0; i < length; i++) {
		if (*string == '\n') {
			setpos(0, software_cursor_y + 1);
			if (software_cursor_y >= height) {
				shiftUp();
			}
			string++;
		}
		else {
			show(software_cursor_x, software_cursor_y, *string, attr);
			setpos(software_cursor_x + 1, software_cursor_y);
			if (software_cursor_x >= width) { // ausserhalb (x-richtgung)
				setpos(0, software_cursor_y + 1);
				
				if (software_cursor_y >= height) {
					shiftUp();
				}

			}
			string++;
		}
	}	
}

void CGA_Screen::shiftUp() {
	setpos(software_cursor_x, height - 1);
	for (int l = 0; l < height; l++) {
		for (int m = 0; m < width; m++) {
			if (l == height - 1) {
				char *act = getAdress(m, l);
				*act = ' ';
			}
			else {
				char *act = getAdress(m, l);
				char *nex = getAdress(m, l+1);
				*act = *nex;
				act++; nex++;
				*act = *nex;
			}
		}
	}
}

char * CGA_Screen::getAdress(int x, int y) {
	return (char *)(startAdress + (x + y*80)*2);
}
