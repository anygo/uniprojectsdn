// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse CGA_Stream
 */

#ifndef __cgastr_include__
#define __cgastr_include__

#include "object/o_stream.h"
#include "machine/cgascr.h"

/*! \brief Darstellung verschiedener Datentypen auf dem Bildschrim
 * 
 *  Die Klasse CGA_Stream ermöglicht die Ausgabe verschiedener Datentypen als
 *  Zeichenketten auf dem CGA Bildschirm eines PCs. Dazu braucht CGA_Stream nur
 *  von den Klassen O_Stream und CGA_Screen abgeleitet und endlich die Methode 
 *  flush() implementiert werden. Für weitergehende Formatierung oder spezielle
 *  Effekte stehen die Methoden der Klasse CGA_Screen zur Verfügung. 
 */
class CGA_Stream : public CGA_Screen, public O_Stream {
private:
	CGA_Stream(CGA_Stream &copy); // Verhindere Kopieren
	unsigned char attr;
public:
	CGA_Stream(int from_col, int to_col, int from_row, int to_row, bool use_cursor = false);
	virtual ~CGA_Stream();
	virtual void flush();
	void setColor(CGA_Screen::color bg, CGA_Screen::color fg, bool blink = false);
	void setColor(unsigned char newattr);
	unsigned char getColor();
	void clearScreen() { CGA_Screen::clearScreen(attr); }
};
#endif
