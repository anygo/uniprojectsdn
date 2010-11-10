// $Date: 2009-09-17 12:48:38 +0200 (Thu, 17 Sep 2009) $, $Revision: 2220 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file 
 * \brief Enthält die Klasse LAPIC zum Zugriff auf den Local APIC 
 *
 *  Local APIC referenziert in Intel System Programming Guide 3A
 */

#ifndef __LAPIC_H__
#define __LAPIC_H__

#include "machine/lapic_registers.h"
/*! \brief Abstraktion des in der CPU integrierten local APICs.
 * 
 *  In modernen PCs besitzt jede CPU einen sogenannten "local APIC". Dieser 
 *  vermittelt zwischen dem I/O APIC, an den die externen Interruptquellen 
 *  angeschlossen sind, und der CPU. Interruptnachrichten, welche den 
 *  lokalen APIC von aussen erreichen, werden an den zugeordneten Prozessorkern
 *  weitergereicht, um dort die Interruptbearbeitung anzustoßen. 
 *  
 *  In Multiprozessorsystem ist es darüberhinaus noch möglich mit Hilfe des
 *  lokalen APICs Nachrichten in Form von Interprozessorinterrupts an andere 
 *  CPUs zu schicken bzw. zu empfangen.  
 */
class LAPIC {
private:
	void write(unsigned short reg, LAPICRegister_t value);
	LAPICRegister_t read(unsigned short reg);

	/// System Programming Guide 3A, p. 9-8 - 9-10
	enum {
		lapicid_reg			= 0x020, // Local APIC ID Register, R/W
		lapicver_reg		= 0x030, // Local APIC Version Register, RO
		tpr_reg				= 0x080, // Task Priority Register, R/W
		eoi_reg				= 0x0b0, // EOI Register, WO
		ldr_reg				= 0x0d0, // Logical Destination Register, R/W
		dfr_reg				= 0x0e0, // Destination Format Register, bits 0-27 RO, bits 28-31 R/W
		spiv_reg			= 0x0f0, // Spurious Interrupt Vector Register, bits 0-8 R/W, bits 9-1 R/W
		icrl_reg			= 0x300, // Interrupt Command Register 1, R/W
		icrh_reg			= 0x310, // Interrupt Command Register 2, R/W
	};

public:
	/// \brief Konstruktor
	LAPIC() {}
	/// \brief Initalisiert den local APIC der jeweiligen CPU
	void init();
	/*! \brief Signalisiert EOI(End of interrupt)
	 *  
	 *  Teilt dem local APIC mit, dass die aktuelle Interruptbehandlung
	 *  abgeschlossen ist. Diese Funktion muss gegen Ende der 
	 *  Unterbrechungsbehandlung aufgerufen werden und zwar bevor 
	 *  prozessorseitig die Unterbrechungen wieder zugelassen werden. 
	 */
	void ackIRQ();
    /*! \brief Liefert die ID des in der aktuellen CPU integrieren APICs
	 *  \return lAPIC ID
	 */
	unsigned char getLAPICID();
	/*! \brief Liefert eindeutige Identifikation der jeweiligen CPU 
	 *  \return lAPIC ID als Identifikation der CPU
	 */
	unsigned char getCPUID() {
		return getLAPICID();
	}
	/*! \brief Liefert Versionsnummer des local APICs
	 *  \return Versionsnummer
	 */
	unsigned char getVersion();
	/*! \brief Setzt die local APIC ID im LDR Register
	 *  \param id APIC ID
	 */
	void setLogicalLAPICID(unsigned char id);
	/*! \brief Verschickt einen IPI an die adressieren CPU(s)
	 *  \param destination Maske mit Zielcpu(s)
	 *  \param data Einstellungen 
	 */
	void sendIPI(unsigned char destination, struct ICR_L data);
	/*! \brief Kehrt mit true zurück, falls zum Zeitpunkt des Aufrufs kein IPI 
	 *  aktiv ist. Kehrt mit false zurück, falls der letzte gesendete IPI noch
	 *  nicht vom Zielprozessor akzeptiert wurde.
	 *  \return 
	 */
	bool isIPIDelivered();
	/// \brief Ist dieser lAPIC extern?
	bool isExternalAPIC();
    /// \brief Ist dieser lAPIC intern?
	bool isLocalAPIC();
	/// \brief Ist diese CPU ein PentiumIV?
	bool isPentium4();
};

// global object declaration
extern LAPIC lapic;

#endif
