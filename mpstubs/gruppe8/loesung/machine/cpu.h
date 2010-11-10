// $Date: 2009-10-02 12:49:23 +0200 (Fri, 02 Oct 2009) $, $Revision: 2229 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält die Klasse CPU
 */

#ifndef __CPU_include__
#define __CPU_include__

/// \brief Assemblerimplementierung, um Interrupts zu aktivieren.
extern "C" void int_enable();
/// \brief Assemblerimplementierung, um Interrupts zu deaktivieren.
extern "C" void int_disable();
/*! \brief Assemblerimplementierung, um atomar Interrupts zu aktivieren und 
 *  in den Idlemodus zu gehen.
 */
extern "C" void cpu_idle();
/// \brief Assemblerimplmentierung, um den Prozessor anzuhalten.
extern "C" void cpu_halt();

/*! \brief Implementierung einer Abstraktion fuer den Prozessor.   
 *  
 *  Derzeit wird nur angeboten, Interrupts zuzulassen, zu verbieten, den 
 *  Prozessor in den Haltmodus zu schicken oder ganz anzuhalten.
 */ 
class CPU {
private:
	CPU(const CPU &copy); // Verhindere Kopieren
public:
	/// \brief Konstruktor
	CPU() {}
	/*! \brief Erlauben von (Hardware-)Interrupts
	 *  
	 *  Lässt die Unterbrechungsbehandlung zu, indem die Assembleranweisung 
	 *  \b sti ausgeführt wird.
	 */
	inline void enable_int() {
		int_enable ();
	}
	/*! \brief Interrupts werden ignoriert/verboten
	 *  
	 *  Verhindert eine Reaktion auf Unterbrechungen, indem die 
	 *  Assembleranweisung \b cli ausgeführt wird.
	 */
	inline void disable_int() {
		int_disable ();
	}
	/*! \brief Prozessor bis zum nÃ¤chsten Interrupt anhalten
	 * 
	 *  Versetzt den Prozessor in den Haltezustand, aus dem er nur durch einen
	 *  Interrupt wieder erwacht. Intern werden dazu die Interrupts mit \b sti 
	 *  freigegeben und der Prozessor mit \b hlt angehalten. Intel garantiert,
	 *  dass die Befehlsfolge \b sti \b hlt atomar ausgeführt wird.
	 */
	inline void idle() {
		cpu_idle ();
	}
	/*! \brief Prozessor anhalten
	 * 
     *  Hält den Prozessor an. Intern werden dazu die Interrupts mit \b cli 
     *  gesperrt und anschließend der Prozessor mit \b hlt angehalten. Da der
     *  Haltezustand nur durch einen Interrupt verlassen werden könnte, ist 
     *  somit garantiert, dass die CPU bis zum nächsten Kaltstart "steht". 
     *  Das Programm kehrt aus halt() nie zurück. In einer 
     *  Multiprozessorumgebung hat die Ausführung des Halt-Befehls nur 
     *  Auswirkungen auf die CPU, die ihn ausführt. Die anderen CPUs laufen
     *  jedoch weiter.
	 */
	inline void halt() {
		cpu_halt ();
	}
};

#endif
