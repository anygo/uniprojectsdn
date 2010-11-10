// $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse Stringbuffer
 */

#ifndef __strbuf_include__
#define __strbuf_include__

/*! \brief Die Klasse Stringbuffer dient dazu, einzelne Zeichen zu längeren Texten 
 *  zusammenzustellen, die dann an einem Stück verarbeitet werden können. 
 *  
 *  Damit ein möglichst vielseitiger Einsatz möglich ist, trifft die Klasse
 *  keine Annahme darüber, was "verarbeiten" in diesem Zusammenhang bedeutet. 
 *  Nur der Zeitpunkt der Verarbeitung steht bereits fest, nämlich immer, 
 *  wenn dies explizit gewünscht wird oder der Text so lang geworden ist, 
 *  dass keine weiteren Zeichen hinzugefügt werden können. Dies geschieht durch
 *  Aufruf der Methode flush(). Da Stringbuffer geräteunabhängig sein soll, 
 *  ist flush() eine virtuelle Methode, die von den abgeleiteten Klassen 
 *  definiert werden muss.
 *  
 *  \par Hinweise zur Implementierung
 *  Zur Pufferung der Zeichen eignet sich ein fest dimensioniertes Feld, auf 
 *  das die abgeleiteten Klassen zugreifen können müssen. Auch die Anzahl der
 *  Zeichen oder das zuletzt beschriebene Feldelement sollte in den 
 *  spezialisierten flush() Methoden erkennbar sein.
 *  
 *  \par Anmerkung
 *  Anlass für die Einführung dieser Klasse war die Erkenntnis, dass Ausgaben
 *  eines Programmes sehr häufig aus vielen kleinen Komponenten bestehen, zum
 *  Beispiel, wenn die Namen und Inhalte von Variablen dargestellt werden 
 *  sollen. Andererseits können wenige längere Texte meist viel effizienter 
 *  ausgegeben werden als viele kurze. Daher erscheint es sinnvoll, vor der 
 *  Ausgabe die einzelnen Komponenten mit Hilfe eines Stringbuffer Objekts 
 *  zusammenzufügen und erst später, z. B. bei einem Zeilenumbruch, gesammelt 
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
