// $Date: 2009-09-25 13:35:09 +0200 (Fri, 25 Sep 2009) $, $Revision: 2223 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file 
 *  \brief Enth�lt die Funktion CPUstartup.
 */

#include "machine/apicsystem.h"

extern "C" int main_ap() __attribute__ ((weak));
extern "C" int main();

/*! \brief Einsprungpunkt ins C/C++ System.
 *  
 *  Die in Assembler geschriebenen Startup-Routinen springen CPUstartup an, 
 *  welches dann die Mainfunktion, je f�r den Bootstrap Processor (BSP) bzw. die
 *  Application Processors (AP) anspringt. 
 *  
 *  \param isBSP Gibt an, ob die aktuelle CPU der Bootstrap Processor ist.
 *  Bei der Uniprozessorl�sung ist dies immer der Fall. main_ap() wird dort 
 *  nat�rlich nicht ben�tigt.
 */
extern "C" void CPUstartup(int isBSP) {
    if(isBSP) {
        system.detectSystemType();
        system.initLAPIC();
        main();
    } else {
        if(main_ap) {
            main_ap();
        }
    }
}
