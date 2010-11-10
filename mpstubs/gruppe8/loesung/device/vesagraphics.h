
/*! \file
 *  \brief Enth�lt die Klasse VESAGraphics.
 */

#ifndef __VESAGRAPHICS_H__
#define __VESAGRAPHICS_H__

#include "machine/vesascreen.h"

/*! \brief Treiber f�r die VESAGrafikkarte.
 * 
 *  Erweitert VESAScreen um die beiden Methoden switch_buffers() und
 *  scanout_frontbuffer(). Mit deren Hilfe ist es m�glich eine Art
 *  Triplebuffering "in Hand" zu implementieren. Da diese beiden Methoden unter
 *  gegenseitigem Ausschluss ausgef�hrt werden m�ssen (w�hrend
 *  scanout_frontbuffer l�uft, soll nicht der Puffer umgeschaltet werden) sind
 *  sie entweder in der guarded Variante oder im Epilog zu verwenden. 
 *  
 *  Die Benutzung des Graphikmodus sieht dann grob ungef�hr so aus:
 *  while(true) {
 *      //Puffer mit Inhalt f�llen
 *	//Umschalten mit switch_buffers()
 *  }
 *  Den Aufruf von scanout_frontbuffer kann man nun auf zwei Arten durchf�hren.
 *  Zum einen ist es m�glich ihn einfach die Schleife zu integrieren. Es ist
 *  aber auch m�glich den Puffer mit einer festen Frequenz neu zu bef�llen.
 *  Dazu ruft man scanout_frontbuffer sinnigerweise im Epilog des Timers auf. 
 */
class VESAGraphics : public VESAScreen {
    void* frontbuffer;    
    bool frontbuffer_new;
public:
    /*! \brief Konstruktor; bekommt zwei Puffer im Hauptspeicher als Parameter
     */
    VESAGraphics(void* frontbuffer, void* backbuffer);
    /*! \brief Tauscht Frontbuffer und Backbuffer aus. Zeichenoperationen �ber
     * die Methoden von VESAScreen gehen immer in den aktuellen Backbuffer;
     * scanout_frontbuffer kopiert immer den aktuellen Frontbuffer in den Speicher
     * der Grafikkarte.
     */
    void switch_buffers();
    /*! \brief Kopiert den aktuellen Frontbuffer in den Speicher der
     * Grafikkarte.
     */
    void scanout_frontbuffer();
};

#endif
