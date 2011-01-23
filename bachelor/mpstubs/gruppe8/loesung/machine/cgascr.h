// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse zum Zugriff auf den CGA_Screen 
 */

#ifndef __screen_include__
#define __screen_include__

/*! \brief Abstraktion des CGA-Textmodus. 
 * 
 *  Mit Hilfe dieser Klasse kann man auf den Bildschirm des PCs zugreifen.    
 *  Der Zugriff erfolgt direkt auf der Hardwareebene, d.h. über den 
 *  Bildschirmspeicher bzw. die I/O-Ports der Grafikkarte. 
 *  
 *  Die Implementierung soll es dabei ermöglichen die Ausgaben des CGA_Screens
 *  nur auf einem Teil des kompletten CGA-Bildschrims darzustellen. Dadurch 
 *  ist es möglich die Ausgaben des Programms und etwaige Debugausgaben auf
 *  dem Bildschrim zu trennen, ohne synchronisieren zu müssen.     
 */

#include "machine/io_port.h"

class CGA_Screen {
private:
	char *startAdress;
	int from_row;
	int to_row;
	int from_col;
	int to_col;
	int width;
	int height;
	bool use_cursor;
	int software_cursor_x;
	int software_cursor_y;
	IO_Port index_reg;
	IO_Port data_reg;
	char *getAdress(int x, int y);
	CGA_Screen(const CGA_Screen &copy); // Verhindere Kopieren
	void shiftUp();

public:
	enum color {
	  BLACK, BLUE, GREEN, CYAN,
	  RED, MAGENTA, BROWN, LIGHT_GREY,
	  DARK_GREY, LIGHT_BLUE, LIGHT_GREEN, LIGHT_CYAN,
	  LIGHT_RED, LIGHT_MAGENTA, YELLOW, WHITE
	};
	
	enum { STD_ATTR = BLACK << 4 | LIGHT_GREY };
	
	enum { ROWS = 25, COLUMNS = 80 };

	struct cga_char {
	char code;
	char attr;
};

	CGA_Screen(int from_col, int to_col, int from_row, int to_row, bool use_cursor = false);
	void setpos(int x, int y);
	void getpos(int &x, int &y);
	void show (int x, int y, char character, unsigned char attrib=STD_ATTR);
	void print(char *string, int length, unsigned char attrib = STD_ATTR);
	void clearScreen(unsigned char attr);


};
#endif

