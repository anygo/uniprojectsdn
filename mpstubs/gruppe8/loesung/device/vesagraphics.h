
/*! \file
 *  \brief Enthält die Klasse VESAGraphics.
 */

#ifndef __VESAGRAPHICS_H__
#define __VESAGRAPHICS_H__

#include "machine/vesascreen.h"

/*! \brief Treiber für die VESAGrafikkarte.
 * 
 *  Erweitert VESAScreen um die beiden Methoden switch_buffers() und
 *  scanout_frontbuffer(). Mit deren Hilfe ist es möglich eine Art
 *  Triplebuffering "in Hand" zu implementieren. Da diese beiden Methoden unter
 *  gegenseitigem Ausschluss ausgeführt werden müssen (während
 *  scanout_frontbuffer läuft, soll nicht der Puffer umgeschaltet werden) sind
 *  sie entweder in der guarded Variante oder im Epilog zu verwenden. 
 *  
 *  Die Benutzung des Graphikmodus sieht dann grob ungefähr so aus:
 *  while(true) {
 *      //Puffer mit Inhalt füllen
 *	//Umschalten mit switch_buffers()
 *  }
 *  Den Aufruf von scanout_frontbuffer kann man nun auf zwei Arten durchführen.
 *  Zum einen ist es möglich ihn einfach die Schleife zu integrieren. Es ist
 *  aber auch möglich den Puffer mit einer festen Frequenz neu zu befüllen.
 *  Dazu ruft man scanout_frontbuffer sinnigerweise im Epilog des Timers auf. 
 */
class VESAGraphics : public VESAScreen {
    void* frontbuffer;    
    bool frontbuffer_new;
public:
    /*! \brief Konstruktor; bekommt zwei Puffer im Hauptspeicher als Parameter
     */
    VESAGraphics(void* frontbuffer, void* backbuffer);
    /*! \brief Tauscht Frontbuffer und Backbuffer aus. Zeichenoperationen über
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
