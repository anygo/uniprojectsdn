#include "machine/vesascreen.h"

#include "machine/cpu.h"

#include "object/debug.h"

extern CPU cpu;

extern "C" int vesa_init_pmstub();
extern "C" int vesa_get_mode_info_pmstub(int mode_number);
extern "C" int vesa_set_mode_pmstub(int mode_number);

extern char __VESA_DETECTION_CODE_START__;
extern char __VESA_DETECTION_CODE_END__;
extern char VBEInfoBlock;
extern char ModeInfoBlock;
extern char rm_start;

static const unsigned int realmode_segment = 0x90000;

GraphicsPrinter_32 printer_32;
GraphicsPrinter_24 printer_24;
GraphicsPrinter_16 printer_16;

VESAScreen::VESAScreen(void* backbuffer) : modes_found(0), current_mode(0),
printer(&printer_32), backbuffer(backbuffer) { 
    //Set Default Graphics printer, to prevent malfunctions when using print 
    //functions without setting a graphics mode before
}

void VESAScreen::change_mode(VBEModeData_t* to) {
    bytes_pp = to->BitsPerPixel/8;	// Unterstützt nur 16, 24 und 32 bpp
    switch(to->BitsPerPixel) {
	case 32:
	    {
		printer = &printer_32;
		break;
	    }
	case 24:
	    {	
		printer = &printer_24;
		break;
	    }
	case 16:
	    {
		printer = &printer_16;
		break;
	    }
    }
    printer->init(to->XResolution, to->YResolution);
    printer->set_lfb(backbuffer);//to->PhysBasePtr);
    current_mode = to;
}

void VESAScreen::init() {
    //Kopiere Realmodecode nach 0x90000
    char* src = &__VESA_DETECTION_CODE_START__;     
    char* dst = reinterpret_cast<char *>(realmode_segment);
    while (src < &__VESA_DETECTION_CODE_END__) {
	*dst = *src;
	src++;
	dst++;
    }
    int ret;
    //Get VBE Controller Information
    if((ret = vesa_init_pmstub()) == 0) {
	DBG << "vesa-init: " << ret << endl;
	DBG << "could not detect VESA VBE" << endl;
	return;
    }
    struct VBEInfoBlock* vbe_info;
    vbe_info = reinterpret_cast<struct VBEInfoBlock*> (0x90000+(&VBEInfoBlock-&rm_start));
    vbe_info->pretty_print();
    //List Modes
    short* modes = reinterpret_cast<short*>(((vbe_info->VideoModePtr >> 16) << 4) + (vbe_info->VideoModePtr & 0xffff));
    
    DBG << "Modes at: " << (void*) modes << endl;
    for(int i = 0; *modes != -1 && i < mode_count; modes++) {
	if(!vesa_get_mode_info_pmstub(*modes)) {
	    DBG << "vesa_get_mode_info failed for mode " << hex << *modes << dec << endl;
	    continue;
	}
	struct ModeInfoBlock* mode_info;
	mode_info = reinterpret_cast<struct ModeInfoBlock*>(0x90000+(&ModeInfoBlock-&rm_start));
	//Kopiere Modusinformation für alle LFB Modi
	if (mode_info->has_LFB() && mode_info->is_supported() && mode_info->is_graphics_mode() && mode_info->BitsPerPixel >= 16) {
	    DBG << "Detected usable VBE Mode: " << hex << *modes;
	    graphic_modes[modes_found] = VBEModeData_t(mode_info, *modes);
	    graphic_modes[modes_found].pretty_print();
	    modes_found++;
	}
    }
}

VBEModeData_t* VESAScreen::find_mode(unsigned int width, unsigned int height, unsigned char bpp) {
    for (int i = 0; i < modes_found; ++i) {
	VBEModeData_t& mode = graphic_modes[i];
	if (mode.XResolution == width && mode.YResolution == height && mode.BitsPerPixel >= bpp) {
	    return &mode; 
	}
    }
    return 0; 
}

bool VESAScreen::set_mode(VBEModeData_t* mode) {
    unsigned short param = 0;
    param |= mode->ModeDesc.modeNumber;
    param |= (1 << 14);
    if (vesa_set_mode_pmstub(param) == 1) {
	change_mode(mode);	
	return true;
    } else {
	return false;
    }
}

