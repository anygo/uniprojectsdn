
/*! \file
 *  \brief Enthält die Klasse Guarded_VESAGraphics.
 */

#ifndef __GUARDED_VESAGRAPHICS_H__
#define __GUARDED_VESAGRAPHICS_H__

#include "device/vesagraphics.h"

#include "guard/secure.h"

/*! \brief Schnittstelle der Anwendung zur Verwendung von Guarded_VESAGraphics
 */
class Guarded_VESAGraphics : public VESAGraphics {
    public:
    Guarded_VESAGraphics(void* frontbuffer, void* backbuffer) : VESAGraphics(frontbuffer, backbuffer) { }
    void switch_buffers() { 
	Secure sec;
	VESAGraphics::switch_buffers();
    }
    void scanout_frontbuffer() {
	Secure sec;
	VESAGraphics::scanout_frontbuffer();
    }
};

#endif
