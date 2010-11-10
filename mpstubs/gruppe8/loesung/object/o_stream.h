// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file 
 *  \brief Hier ist die Klasse O_Stream implementiert. 
 *  Neben der Klasse O_Stream sind hier auch die Manipulatoren hex, dec, oct 
 *  und bin f�r die Wahl der Basis bei der Zahlendarstellung, sowie endl    
 *  f�r den Zeilenumbruch definiert. 
 *  
 *  \par Manipulatoren
 *  Um bei der Textformatierung mit Hilfe der Klasse O_Stream das Zahlensystem
 *  bequem w�hlen und Zeilenumbr�che einf�gen zu k�nnen, sollen sogenannte
 *  Manipulatoren definiert werden. 
 *  Der Ausdruck <b>kout << "a = " << dec << a << " ist hexadezimal " << hex << a << endl;</b>
 *  soll dann beispielsweise den Wert der Variablen a erst in dezimaler und 
 *  dann in hexadezimaler Schreibweise formatieren und zum Schluss einen
 *  Zeilenumbruch anf�gen. 
 *  
 *  Die gew�nschten Eigenschaften k�nnen realisiert werden, wenn hex, dec,
 *  oct, bin und endl als Funktionen (d.h. nicht als Methoden der Klasse 
 *  O_Stream) definiert werden, die als Parameter und R�ckgabewert jeweils 
 *  eine Referenz auf ein O_Stream Objekt erhalten bzw. liefern. Durch diese
 *  Signatur wird bei dem genannten Ausdruck der bereits erw�hnte Operator
 *  O_Stream& O_Stream::operator<< ((*fkt*) (O_Stream&)) ausgew�hlt, der dann
 *  nur noch die als Parameter angegebene Funktion ausf�hren muss.
 *  
 *  \par Anmerkung
 *  Der Manipulatorbegriff wurde dem Buch 
 *  <a href="http://ivs.cs.uni-magdeburg.de/bs/service/buecher/cc.shtml#Stroustrup">"Bjarne Stroustrup: The C++ Programming Language"</a> entnommen.
 *  Dort finden sich auch weitergehende Erl�uterungen dazu.  
 */

#ifndef __o_stream_include__
#define __o_stream_include__

#include "object/strbuf.h"

/*! \brief Die Aufgaben der Klasse O_Stream entsprechen im Wesentlichen denen der 
 *  Klasse ostream der bekannten C++ IO-Streams-Bibliothek.
 * 
 *  Da die Methode put(char) der Basisklasse Stringbuffer recht unbequem ist,
 *  wenn die zusammenzustellenden Texte nicht nur aus einzelnen Zeichen, sondern
 *  auch aus Zahlen, oder selbst wieder aus Zeichenketten bestehen, werden in der
 *  Klasse O_Stream M�glichkeiten zum Zusammenstellen verschiedener Datentypen 
 *  realisiert. In Anlehnung an die bekannten Ausgabeoperatoren der 
 *  C++ IO-Streams-Bibliothek wird dazu der Shift-Operator operator<< verwendet.
 * 
 *  Dar�berhinaus soll es m�glich sein, f�r die Darstellung ganzer Zahlen 
 *  zwischen dem Dezimal-, dem Bin�r- dem Oktal- und dem Hexadezimalsystem 
 *  zu w�hlen. Beachtet dabei bitte die �bliche Darstellung negativer Zahlen: 
 *  Im Dezimalsystem mit f�hrendem Minuszeichen, im Oktal- und Hexadezimalsystem 
 *  ohne Minuszeichen, sondern genau so wie sie im Maschinenwort stehen. 
 *  (Intel-CPUs verwenden intern das 2er-Komplement f�r negative Zahlen. 
 *  -1 ist Hexadeziamal also FFFFFFFF und Oktal 37777777777.) 
 * 
 *  Die �ffentlichen Methoden/Operatoren von O_Stream liefern jeweils eine 
 *  Referenz auf ihr eigenes O_Stream Objekt zur�ck. Dadurch ist es m�glich, 
 *  in einem Ausdruck mehrere der Operatoren zu verwenden, z. B. 
 *  <b>kout << "a = " << a</b>;
 * 
 *  Zur Zeit wird die Darstellung von Zeichen, Zeichenketten und ganzen Zahlen 
 *  unterst�tzt. Ein weiterer << Operator erlaubt die Verwendung von Manipulatoren.
 */

class O_Stream : public Stringbuffer
{
private:
    O_Stream(const O_Stream &copy); // Verhindere Kopieren
    char tmp[128];

public:
	int base;
	O_Stream();
	virtual ~O_Stream();
	virtual void flush()=0;
	O_Stream & operator<<(char c);
	O_Stream & operator<<(unsigned char c);
	O_Stream & operator<<(char *string);
	O_Stream & operator<<(short ival);
	O_Stream & operator<<(unsigned short ival);
	O_Stream & operator<<(int ival);
	O_Stream & operator<<(unsigned int ival);
	O_Stream & operator<<(long ival);
	O_Stream & operator<<(unsigned long ival);
	O_Stream & operator<<(void *ptr);
	O_Stream & operator<<(O_Stream &(*f)(O_Stream &));

};

	O_Stream &hex(O_Stream &os);
	O_Stream &dec(O_Stream &os);
	O_Stream &oct(O_Stream &os);
	O_Stream &bin(O_Stream &os);
	O_Stream &endl(O_Stream &os);

#endif

