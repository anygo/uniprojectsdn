#ifndef __VESADATA_H__
#define __VESADATA_H__

struct VBEInfoBlock {
    char	    VBESignature[4];
    unsigned short  VBEVersion;
    unsigned int    OEMStringPtr;
    char	    Capabilities[4];
    unsigned int    VideoModePtr;
    unsigned short  TotalMemory;
    unsigned short  OEMSoftwareRev;
    unsigned int    OEMVendorNamePtr;
    unsigned int    OEMProductNamePtr;
    unsigned int    OEMProductRevPtr;
    char	    resereved[222];
    char	    OEMData[256];
    void pretty_print() {
	DBG << "VBEInfoBlock" << endl;
	DBG << "Signature: " << VBESignature[0] << VBESignature[1] << VBESignature[2] << VBESignature[3] << endl;
	DBG << "Version: " << hex << VBEVersion << endl;
	DBG << "VideoModePtr: " << hex << (VideoModePtr >> 16) << ":" << (VideoModePtr & 0xffff) << endl;
	DBG << "TotalMemory: " << dec << TotalMemory << endl;
	DBG << dec << endl;
    }

}__attribute__((packed));

struct ModeAttrib {
    unsigned short  supported_by_hardware	    : 1,
		    reserved_1			    : 1,
		    tty_output_supported	    : 1,
		    color			    : 1,
		    graphics_mode		    : 1,
		    vga_compatible		    : 1,
		    vga_compatible_windowed	    : 1,
		    linear_framebuffer_available    : 1,
		    reserved_2			    : 8;

}__attribute__((packed));

typedef struct ModeAttrib VBEModeAttributes;

struct ModeInfoBlock {
    VBEModeAttributes ModeAttributes;
    unsigned char     WinAAttributes;
    unsigned char     WinBAttributes;
    unsigned short    WinGranularity;
    unsigned short    WinSize;
    unsigned short    WinASegment;
    unsigned short    WinBSegment;
    unsigned int      WinFuncPtr;
    unsigned short    BytesPerScanLine;

    unsigned short    XResolution;
    unsigned short    YResolution;
    unsigned char     XCharSize;
    unsigned char     YCharSize;
    unsigned char     NumberOfPlanes;
    unsigned char     BitsPerPixel;
    unsigned char     NumberOfBanks;
    unsigned char     MemoryModel;
    unsigned char     BankSize;
    unsigned char     NumberOfImagePages;
    unsigned char     Reserved_1;

    unsigned char     RedMaskSize;
    unsigned char     RedFieldPosition;
    unsigned char     GreenMaskSize;
    unsigned char     GreenFieldPosition;
    unsigned char     BlueMaskSize;
    unsigned char     BlueFieldPosition;
    unsigned char     RsvdMaskSize;
    unsigned char     RsvdFieldPosition;
    unsigned char     DirectColorModeInfo;
    
    void*	      PhysBasePtr;
    void*	      OffScreenMemOffset;
    unsigned short    OffScreenMemSize;
    unsigned char     Reserved_2[206];

    inline bool has_LFB() {
	return ModeAttributes.linear_framebuffer_available == 1;
    }
    inline bool is_supported() {
	return ModeAttributes.supported_by_hardware == 1;
    }
    inline bool is_graphics_mode() {
	return ModeAttributes.graphics_mode == 1;
    }
}__attribute__((packed));

struct VBEMode {
    unsigned short  modeNumber	  : 9,
		    reserved	  : 5,
		    useLinear	  : 1,
		    dontClear	  : 1;
    VBEMode(unsigned short mode_number, bool clear_screen)
	 : modeNumber(mode_number), useLinear(1), dontClear(!clear_screen) { }
    VBEMode() : modeNumber(0), useLinear(1), dontClear(false) { }
}__attribute__((packed));

typedef struct VBEMode VBEMode_t;

struct VBEModeData {
    VBEModeAttributes ModeAttributes;
    unsigned short    XResolution;
    unsigned short    YResolution;
    unsigned char     BitsPerPixel;
    unsigned char     MemoryModel;

    unsigned char     RedMaskSize;
    unsigned char     RedFieldPosition;
    unsigned char     GreenMaskSize;
    unsigned char     GreenFieldPosition;
    unsigned char     BlueMaskSize;
    unsigned char     BlueFieldPosition;
    unsigned char     RsvdMaskSize;
    unsigned char     RsvdFieldPosition;
    unsigned char     DirectColorModeInfo;

    void*	      PhysBasePtr;
    VBEMode_t	      ModeDesc;

    VBEModeData() {}
    VBEModeData(struct ModeInfoBlock* mode, unsigned short mode_number) :
	ModeAttributes(mode->ModeAttributes), XResolution(mode->XResolution),
	YResolution(mode->YResolution), BitsPerPixel(mode->BitsPerPixel),
	MemoryModel(mode->MemoryModel), RedMaskSize(mode->RedMaskSize),
	RedFieldPosition(mode->RedFieldPosition), GreenMaskSize(mode->GreenMaskSize),
	GreenFieldPosition(mode->GreenFieldPosition), BlueMaskSize(mode->BlueMaskSize),
	BlueFieldPosition(mode->BlueFieldPosition), RsvdMaskSize(mode->RsvdMaskSize),
	RsvdFieldPosition(mode->RsvdFieldPosition),
	DirectColorModeInfo(mode->DirectColorModeInfo), PhysBasePtr(mode->PhysBasePtr), 
	ModeDesc(VBEMode_t(mode_number, true)) { }

    void pretty_print() {
	DBG << "Resolution: " << dec << XResolution << 'x' << YResolution << 'x' << static_cast<int>(BitsPerPixel) << "bpp " 
	    << "LFB: " << (ModeAttributes.linear_framebuffer_available ? ((char*) "yes") : ((char*) "no")) << endl;
	DBG << "ColorMode: " << hex << (int) MemoryModel << dec << ", Red: " << (int)RedMaskSize << "bits, Green: " << (int)GreenMaskSize << "bits, Blue: " << (int)BlueMaskSize << "bits, Rsvd: " << (int)RsvdMaskSize << endl;	
	DBG << "Color Layout: " << "Red Pos.: " << (int)RedFieldPosition << ", Green Pos.: " << (int)GreenFieldPosition << ", Blue Pos.: " << (int) BlueFieldPosition << ", RsvdPos.:" << (int)RsvdFieldPosition << endl;
//	DBG << "LFB pos: " << PhysBasePtr << " Mode: " << hex << ModeDesc.modeNumber << dec << endl;
    }

    };

typedef struct VBEModeData VBEModeData_t;

#endif
