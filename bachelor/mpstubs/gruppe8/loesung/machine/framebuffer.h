
/*! \file
 *  \brief Enthält die Klasse Framebuffer.
 */

#ifndef __FRAMEBUFFER_H__
#define __FRAMEBUFFER_H__

#include "object/graphicstypes.h"

template <int width>
class Pixel {
    public:
    unsigned int data : width;
    public:
    Pixel(unsigned int data) : data(data) { } 
}__attribute__((packed));

/*! \brief Die Klasse Framebuffer kapselt die grundlegende Funktionalität, um
 *  den Inhalt eines Stück Speichers als Bitmap zu behandeln.
 */
template <unsigned char pixel_width, unsigned char red_offset, unsigned char green_offset, unsigned char blue_offset, unsigned char red_size, unsigned char green_size, unsigned char blue_size>
class Framebuffer {
    public:
	typedef Pixel<pixel_width> Pixel_t;
    protected:
	unsigned int x_max;
	unsigned int y_max;
    private:
	Pixel_t* lfb;
    public:
	void init(unsigned int x_max, unsigned int y_max) {
	    this->x_max = x_max;
	    this->y_max = y_max;
	    put_pixel(100, 100, 0, 128, 0); 
	}
    protected:
	void set_lfb(void* lfb) {
	    this->lfb = reinterpret_cast<Pixel_t*>(lfb); 
	}
	inline void put_pixel(void* pos, const unsigned char red, const unsigned char green, const unsigned char blue) {
	    Pixel_t pixel((red << red_offset) | (green << green_offset) | (blue << blue_offset));
	    *reinterpret_cast<Pixel_t*>(pos) = pixel;
	}
	inline void put_pixel(Pixel_t* pos, const Color& color) {
	    put_pixel(pos, color.red, color.green, color.blue);
	}
	inline void put_pixel(const unsigned int x, const unsigned int y, const unsigned char red, const unsigned char green, const unsigned char blue) {
	    put_pixel(lfb + (y * x_max + x), red, green, blue);
	}
	inline Color get_pixel(Pixel_t* pos) {
	    Pixel_t pix = *reinterpret_cast<Pixel_t*>(pos);
	    Color col;
	    col.red =  (pix.data >> red_offset) & ((1 << red_size) - 1);
	    col.green =  (pix.data >> green_offset) & ((1 << green_size) - 1);
	    col.blue =  (pix.data >> blue_offset) & ((1 << blue_size) - 1);
	    return col;
	}
	inline Pixel_t* get_pointer(const Point& p) {
	    return lfb + (p.x + p.y * x_max);
	}
	inline void blit_bitmap(const Point& p, unsigned int width, unsigned int height, void* bitmap, const Color& color) {
	    unsigned short width_byte = width/8 + ((width%8 != 0) ? 1 : 0);
	    char* sprite = reinterpret_cast<char*>(bitmap);
	    for(unsigned int y = 0; y < height; ++y) {
		Pixel_t* pixel = reinterpret_cast<Pixel_t*>(this->get_pointer(p)) + y * this->x_max;
		for(unsigned int x = 0; x < width_byte; ++x) {
		    for(int src = 7; src >= 0; --src) {
			if ((1 << src) & *sprite) {
			    this->put_pixel(pixel, color);
			}
			pixel++;
		    }
		    sprite++;
		}
	    }
	}
};


#endif
