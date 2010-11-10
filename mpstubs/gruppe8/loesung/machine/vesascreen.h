
/*! \file
 *  \brief Enthält die Klasse VESAScreen.
 */

#ifndef __VESASCREEN_H__
#define __VESASCREEN_H__

#include "object/debug.h"
#include "object/graphicsprinter.h"
#include "machine/vesadata.h"

static const int mode_count = 20;

/*! \brief Abstraktion einer VESA Grafikkarte
 *  
 *  Die Klasse VESAScreen bietet die Möglichkeit VESA Grafikmodi zu setzen. In
 *  diesen kann man dann direkt den in den Adressraum gemappten Framebuffer
 *  schreiben. Die Klasse GraphicsPrinter wird dann dazu verwendet, um einige
 *  primitive Zeichenoperationen durchführen zu können. 
 *  
 *  Die Verwendung sieht folgendermassen aus:
 *  Zuerst muss die Methode VESAScreen::init() aufgerufen werden, um
 *  Informationen, über die von der Grafikkarte unterstützen Modi zu erhalten.
 *  Danach kann man mit VESAScreen::find_mode() nach einem geeigneten Modus
 *  suchen. (Je nach Grafikkarte kann es auch passieren, dass kein passender
 *  Modus gefunden wird). Mit Aufruf von VESAScreen::set_mode() kann man dann
 *  den vorher gefundenen Modus setzen. Wenn ihr eine Übersicht über die von
 *  der aktuellen Grafikkarte unterstützen Modi haben wollt, dann ruft einfach
 *  VESAScreen::init() auf, ohne dann nachher in den Grafikmodus umzuschalten.
 */
class VESAScreen {
    private:
	VBEModeData_t graphic_modes[mode_count];
	int modes_found;
    protected:
	VBEModeData_t* current_mode;
	AbstractGraphicsPrinter* printer;
	unsigned char bytes_pp;
	void* lfb;
	void* backbuffer;
    private:
	void change_mode(VBEModeData_t* to);
    public:

	VESAScreen(void* backbuffer);
	/*! \brief Initalisiert das Grafiksubsystem; Aufruf am besten in der Main-Funktion
	 */
	void init(); 
	/*! \brief Sucht einen Modus aus der Modustabelle
	 *  
	 *  Nachdem init() ausgeführt wurde, kann man die von der Grafikkarte
	 *  unterstützten Modi nach gewissen Kritieren durchsuchen, um einen
	 *  geeigneten Modus zu finden.
	 *  \param width Breite in Pixel des gewünschen Grafikmodus
	 *  \param height Höhe in Pixel des gewünschen Grafikmodus
	 *  \param bpp Untere Schranke für die Farbtiefe des gewünschten
	 *  Grafikmodus (Achtung: qemu kann kein 32bpp, sondern nur 24bpp, bei
	 *  den Testrechnern ist dies genau invers(32bpp aber kein 24bpp).
	 *  \return Modus, der am Besten zu den gewählten Parametern passt. 
	 */
	VBEModeData_t* find_mode(unsigned int width, unsigned int height, unsigned char bpp);
	/*! \brief Setzt einen vorher per find_mode ausgewählten Modus
	 *  \param Zeiger auf den Modusdeskriptor
	 */
	bool set_mode(VBEModeData_t* mode);
	/*! \brief Setzt sämtliche Pixel im aktuellen Puffer auf schwarz
	 */
	inline void clear_screen() {
	    printer->clear_screen();   
	}
	/*! \brief Zeichnet eine Linie von \b start nach \b end
	 *  \param start Startpunkt der Linie
	 *  \param end Endpunkt der Linie
	 */
	inline void print_line(const Point& start, const Point& end, const Color& color) {
	    printer->print_line(start,end,color);
	}
	/*! \brief Zeichnet ein Rechteck
	 *  \param top_left Obere, linke Ecke des Rechtecks
	 *  \param bottom_right Untere, rechte Ecke des Rechtecks
	 *  \color Farbe, in der das Rechteck gezeichnet werden soll
	 *  \filled Gibt an, ob das Rechteck gefüllt gezeichnet werden soll,
	 *  oder nur als Rahmen
	 */
	inline void print_rectangle(const Point& top_left, const Point& bottom_right, const Color& color, bool filled = true) {
	    printer->print_rectangle(top_left, bottom_right, color, filled);
	}
	/*! \brief Ändern der Schriftart für Textausgabe im Grafikmodus
	 *  \param new_font Schriftart, die bei nachfolgenden Aufrufen von print_text verwendet werden soll.
	 */
	inline void set_font(const Font& new_font) {
	    printer->set_font(new_font);
	}
	/*! \brief Gibt Text an der globalen Cursorposition (analog CGA_Screen)
	 *  auch mit Zeilenumbruch aus(allerdings ohne scrollen.
	 *  \param string Zeiger auf den String, der ausgegeben werden soll
	 *  \param len Länge des auszugebenden Strings
	 *  \param color Farbe in der der String ausgegeben werden soll
	 */
	inline void print_text(char* string, int len, const Color& color) {
	    printer->print_text(string, len, color);
	}
	/*! \brief Ausgabe von Text an der Position \b pos (ohne automatischen Zeilenumbruch)
	 *  \param string Zeiger auf den String, der ausgegeben werden soll
	 *  \param len Länge des auszugebenden Strings
	 *  \param color Farbe in der der String ausgegeben werden soll
	 */
	inline void print_text(char* string, int len, const Color& color, const Point& pos) {
	    printer->print_text(string, len, color, pos);
	}
	/*! \brief Ausgabe eines Sprites mit Alpha-Blending.
	 *  
	 *  Gibt eine Spritebitmap aus, und überblendet sie mit Hilfe von
	 *  Alpha-blending anhand des Alpha-Kanals mit dem schon im Framebuffer
	 *  vorhanden Hintergrund. Das Layout eines Pixels ist RGBA.
	 *  Funktioniert bis jetzt nur in 24/32bpp Modi. In GIMP lassen sich
	 *  Bitmaps als C-Source exportieren. Diese kann mann dann hiermit
	 *  verwenden.
	 *  \param p Linke, obere Ecke des Sprites auf den Bildschrim
         *  \param sprite_width Breite des Sprites
         *  \param sprite_height Höhe des Sprites
	 *  \param sprite Zeiger auf die Binärdaten des Sprites
	 */
	inline void print_sprite_alpha(const Point& p, int sprite_width, int sprite_height, const SpritePixel* sprite) {
	    printer->print_sprite_alpha(p, sprite_width, sprite_height, sprite);
	}
};


#endif
