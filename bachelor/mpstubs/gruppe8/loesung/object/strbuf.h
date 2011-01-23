// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enth�lt die Klasse Stringbuffer
 */

#ifndef __strbuf_include__
#define __strbuf_include__

/*! \brief Die Klasse Stringbuffer dient dazu, einzelne Zeichen zu l�ngeren Texten 
 *  zusammenzustellen, die dann an einem St�ck verarbeitet werden k�nnen. 
 *  
 *  Damit ein m�glichst vielseitiger Einsatz m�glich ist, trifft die Klasse
 *  keine Annahme dar�ber, was "verarbeiten" in diesem Zusammenhang bedeutet. 
 *  Nur der Zeitpunkt der Verarbeitung steht bereits fest, n�mlich immer, 
 *  wenn dies explizit gew�nscht wird oder der Text so lang geworden ist, 
 *  dass keine weiteren Zeichen hinzugef�gt werden k�nnen. Dies geschieht durch
 *  Aufruf der Methode flush(). Da Stringbuffer ger�teunabh�ngig sein soll, 
 *  ist flush() eine virtuelle Methode, die von den abgeleiteten Klassen 
 *  definiert werden muss.
 *  
 *  \par Hinweise zur Implementierung
 *  Zur Pufferung der Zeichen eignet sich ein fest dimensioniertes Feld, auf 
 *  das die abgeleiteten Klassen zugreifen k�nnen m�ssen. Auch die Anzahl der
 *  Zeichen oder das zuletzt beschriebene Feldelement sollte in den 
 *  spezialisierten flush() Methoden erkennbar sein.
 *  
 *  \par Anmerkung
 *  Anlass f�r die Einf�hrung dieser Klasse war die Erkenntnis, dass Ausgaben
 *  eines Programmes sehr h�ufig aus vielen kleinen Komponenten bestehen, zum
 *  Beispiel, wenn die Namen und Inhalte von Variablen dargestellt werden 
 *  sollen. Andererseits k�nnen wenige l�ngere Texte meist viel effizienter 
 *  ausgegeben werden als viele kurze. Daher erscheint es sinnvoll, vor der 
 *  Ausgabe die einzelnen Komponenten mit Hilfe eines Stringbuffer Objekts 
 *  zusammenzuf�gen und erst sp�ter, z. B. bei einem Zeilenumbruch, gesammelt 
 *  auszugeben. 
 */
class Stringbuffer {
private:
    Stringbuffer(const Stringbuffer &copy); // Verhindere Kopieren
/* Hier muesst ihr selbst Code vervollstaendigen */     


protected: 
	Stringbuffer();
	virtual ~Stringbuffer();
	void put(char c);
	virtual void flush()=0;

	char buffer[80];
	int pos;
};

#endif
