#include "device/vesagraphics.h"

#include "machine/memcpy32.h"

VESAGraphics::VESAGraphics(void* frontbuffer, void* backbuffer) 
	: VESAScreen(backbuffer), frontbuffer(frontbuffer), frontbuffer_new(false) {
}

void VESAGraphics::switch_buffers() {
    void* new_front = backbuffer; 
    backbuffer = frontbuffer;
    frontbuffer = new_front;
    printer->set_lfb(backbuffer);
    frontbuffer_new = true;
}

void VESAGraphics::scanout_frontbuffer() {
    if(frontbuffer_new) {
	memcpy32(bytes_pp * current_mode->XResolution * current_mode->YResolution, frontbuffer, current_mode->PhysBasePtr); 
    }
    frontbuffer_new = false;
}

