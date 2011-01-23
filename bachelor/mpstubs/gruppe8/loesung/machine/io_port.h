// $Date: 2009-09-15 18:16:55 +0200 (Tue, 15 Sep 2009) $, $Revision: 2217 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Zugriffsklasse für den IO-Adressraum des x86
 *  
 *  Die Funktionen, die zum Lesen und Schreiben auf IO-Ports verwendet werden,
 *  sind also Assemblerfunktionen in der Datei machine/io_port.asm implementiert. 
 */

#ifndef __io_port_include__
#define __io_port_include__

/* BENUTZTE FUNKTIONEN */

/*! \brief Assemberimplementierung zum Schreiben eines Bytes auf einem IO-Port
 *  \param port IO-Port Adresse
 *  \param value Zu schreibender Wert
 */
extern "C" void outb(int port, int value);

/*! \brief Assemberimplementierung zum Schreiben eines Wortes auf einem IO-Port
 *  \param port IO-Port Adresse
 *  \param value Zu schreibender Wert
 */
extern "C" void outw(int port, int value);

/*! \brief Assemberimplementierung zum Lesen eines Bytes von einem IO-Port
 *  \param port IO-Port Adresse
 *  \return Gelesener Wert
 */
extern "C" int inb(int port);

/*! \brief Assemberimplementierung zum Lesen eines Wortes von einem IO-Port
 *  \param port IO-Port Adresse
 *  \return Gelesener Wert
 */
extern "C" int inw(int port);

/* KLASSENDEFINITION */

/*!  \brief Die IO_Port-Klasse dient dem Zugriff auf die Ein-/Ausgabeports des PC.
 *   
 *   Beim PC gibt es einen gesonderten I/O-Adressraum, der nur mittels der 
 *   Maschineninstruktionen 'in' und 'out' angesprochen werden kann. 
 *   Ein IO_Port-Objekt wird beim Erstellen an eine Adresse des I/O-Adressraums 
 *   gebunden und kann dann fuer byte- oder wortweise Ein- oder Ausgaben verwendet 
 *   werden. 
 */

class IO_Port {
    // Kopieren erlaubt!
    /*! \brief Adresse im I/O-Adressraum */
    int address;
public:
    /*! \brief Konstruktor
     *  \param a Adresse des IO-Ports im IO-Adressraum
     */ 
    IO_Port(int a) : address (a) {};
    /*! \brief Byteweise Ausgabe eines Wertes ueber einen I/O-Port.
     *  \param val Wert, der ausgegeben werden soll.
     */
    void outb(int val) const { 
        ::outb (address, val); 
    };
    /*! \brief Wortweise Ausgabe eines Wertes ueber einen I/O-Port.
     *  \param val Wert, der ausgegeben werden soll.
     */
    void outw(int val) const { 
        ::outw (address, val); 
    };
    /*! \brief Byteweises Einlesen eines Wertes ueber einen I/O-Port.
     *  \return Gelesenes Byte.
     */
    int inb() const { 
        return ::inb (address); 
    };
    /*! \brief Wortweises Einlesen eines Wertes ueber einen I/O-Port.
     *  \return Gelesenes Wort.
     */    
    int inw() const { 
        return ::inw (address); 
    };
};

#endif
