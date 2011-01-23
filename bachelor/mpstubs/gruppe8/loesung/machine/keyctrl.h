// $Date: 2009-08-14 16:19:58 +0200 (Fri, 14 Aug 2009) $, $Revision: 2209 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält Klasse Keyboard_Controller
 */

#ifndef __Keyboard_Controller_include__
#define __Keyboard_Controller_include__

#include "machine/io_port.h"
#include "machine/key.h"

/*! \brief Abstraktion für den Tastaturcontroller des PCs
 * 
 *  Die Klasse Keyboard_Controller dient dazu, die PC Tastatur zu initialisieren
 *  und aus den gesendeten Make- und Break-Codes den Scan- und ASCII Code der 
 *  gedrückten Taste zu bestimmen. 
 */
class Keyboard_Controller {
private:
    Keyboard_Controller(const Keyboard_Controller &copy); // Verhindere Kopieren
private:
    unsigned char code;
    unsigned char prefix;
    Key gather;
    char leds;

    // Benutzte Ports des Tastaturcontrollers
    const IO_Port ctrl_port; // Status- (R) u. Steuerregister (W)
    const IO_Port data_port; // Ausgabe- (R) u. Eingabepuffer (W)

    /// Bits im Statusregister
    enum { outb = 0x01, inpb = 0x02, auxb = 0x20 };

    /// Kommandos an die Tastatur
    struct kbd_cmd
    {
        enum { set_led = 0xed, set_speed = 0xf3 };
    };
    enum { cpu_reset = 0xfe };

    /// Antworten der Tastatur
    struct kbd_reply
    {
        enum { ack = 0xfa };
    };

    /// Konstanten fuer die Tastaturdekodierung
    enum { break_bit = 0x80, prefix1 = 0xe0, prefix2   = 0xe1 };

    static unsigned char normal_tab[];
    static unsigned char shift_tab[];
    static unsigned char alt_tab[];
    static unsigned char asc_num_tab[];
    static unsigned char scan_num_tab[];

    /*! \brief Interpretiert die Make und Break-Codes der Tastatur und
     *  liefert den ASCII Code, den Scancode und Informationen darüber,
     *  welche zusätzlichen Tasten wie Shift und Ctrl gedrückt wurden. 
     *              
     *  \return true bedeutet, dass das Zeichen komplett ist, anderenfalls
     *  fehlen noch Make- oder Breakcodes.
     */
    bool key_decoded ();

    /*! \brief ermittelt anhand von Tabellen aus dem Scancode und den 
     *  gesetzten Modifier-Bits den ASCII Code der Taste.
     */
    void get_ascii_code ();
public:

    /*! \brief Konstruktor; Initialisierung der Tastatur.
     * 
     *  Alle LEDs werden ausgeschaltet und die Wiederholungsrate auf maximale 
     *  Geschwindigkeit eingestellt.
     */
    Keyboard_Controller ();

    /*! \brief Dient der Tastaturabfrage nach dem Auftreten einer Tastatur-
     *  unterbrechung. 
     *  
     *  Wenn der Tastendruck abgeschlossen ist und ein Scancode,
     *  sowie gegebenenfalls ein ASCII Code emittelt werden konnte, werden diese
     *  in Key zurückgeliefert. Wenn dagegen bloß eine der Spezialtasten Shift, 
     *  Alt, CapsLock usw. gedrückt wurde, dann liefert key_hit() einen ungültigen
     *  Wert zurück, was mit Key::valid() überprüft werden kann.
     *  \return Dekodierte Taste
     */
    Key key_hit ();

    /*! \brief Führt einen Neustart des Rechners durch. Ja, beim PC macht das der
     *  Tastaturcontroller.
     */
    void reboot ();

    /*! \brief Funktion zum Einstellen der Wiederholungsrate der Tastatur.
     * \param delay bestimmt, wie lange eine Taste gedrückt werden muss, bevor 
     * die Wiederholung einsetzt. Erlaubt sind Werte von 0 (250ms), 1 (500ms), 
     * 2 (750ms) und 3(1000ms).
     * \param speed bestimmt, wie schnell die Tastencodes aufeinander folgen 
     * soll. Erlaubt sind Werte zwischen 0 (30 Zeichen pro Sekunde) und 
     * 31 (2 Zeichen pro Sekunde).
     */
    void set_repeat_rate (int speed, int delay);

    /*! \brief Setzt oder löscht die angegebene Leuchtdiode.
     *  \param led Gibt an, welche LED geschaltet werden soll.
     *  \param on LED an- oder ausschalten.
     */ 
    void set_led (char led, bool on);
    
    /// Namen der LEDs
    struct led {
        enum { 
            caps_lock = 4,      ///< Caps Lock
            num_lock = 2,       ///< Num Lock
            scroll_lock = 1     ///< Scroll Lock
        };
    };
};

#endif
