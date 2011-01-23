
/*! \file
 *  \brief Enthält die Klasse GraphicsPrinter.
 */

#ifndef _GRAPHICSPRINTER_H__
#define _GRAPHICSPRINTER_H__

#include "machine/framebuffer.h"
#include "object/graphicstypes.h"
#include "object/fonts/fonts.h"
#include "machine/memcpy32.h"

int abs(int a);

class AbstractGraphicsPrinter {
    public:
    virtual void init(unsigned int x_max, unsigned int y_max) = 0;
    virtual void clear_screen() = 0;
    virtual void print_line(const Point& start, const Point& end, const Color& color) = 0;
    virtual void print_rectangle(const Point& top_left, const Point& bottom_right, const Color& color, bool filled = true) = 0;
    virtual void set_font(const Font& new_font) = 0;
    virtual void print_text(char* string, int len, const Color& color) = 0;
    virtual void print_text(char* string, int len, const Color& color, const Point& pos) = 0;
    virtual void print_sprite_alpha(const Point& p, int sprite_width, int sprite_height, const SpritePixel* sprite) = 0;
    virtual void set_lfb(void* lfb) = 0;
};

/*! \brief GraphicsPrinter implementiert die Zeichenmethoden, die von
 *  VESAScreen und schlussendlich VESAGraphics angeboten werden.
 */
template <unsigned int pixel_width, unsigned char red_offset, unsigned char green_offset, unsigned char blue_offset, unsigned char red_size, unsigned char green_size, unsigned char blue_size>
class GraphicsPrinter : public Framebuffer<pixel_width, red_offset, green_offset, blue_offset, red_size, green_size, blue_size>, public AbstractGraphicsPrinter {
    typedef Framebuffer<pixel_width, red_offset, green_offset, blue_offset, red_size, green_size, blue_size> FB_Base;
    Point cursor;
    Font const * font;
    public:
	typedef Pixel<pixel_width> Pixel_t;
	GraphicsPrinter() : cursor(Point(0, 0)) {}
	void init(unsigned int x_max, unsigned int y_max) {
	    FB_Base::init(x_max, y_max);
	    this->set_font(sun_font_12x22);
	}
	void set_lfb(void* lfb) {
	    FB_Base::set_lfb(lfb);
	}
	void clear_screen() {
	    Pixel_t* cur = this->get_pointer(Point(0,0));
	    memset32(this->x_max * this->y_max * sizeof(Pixel_t), cur, 0);
	}
	void print_line(const Point& start, const Point& end, const Color& color) {
	    int x, y, D, DE, DNE, d_x, d_y, x_i1, x_i2, y_i1, y_i2, steps;
	    d_x = abs(end.x - start.x);
	    d_y = abs(end.y - start.y);
	    if (d_x >= d_y) {
		steps = d_x + 1;
		D = 2 * d_y - d_x;
		DE = d_y << 1;
		DNE = (d_y - d_x) << 1;
		x_i1 = 1;
		x_i2 = 1;
		y_i1 = 0;
		y_i2 = 1;
	    } else {
		steps = d_y + 1;
		D = 2 * d_x - d_y;
		DE = d_x << 1;
		DNE = (d_x - d_y) << 1;
		x_i1 = 0;
		x_i2 = 1;
		y_i1 = 1;
		y_i2 = 1;
	    }
	    if (start.x > end.x) {
		x_i1 = -x_i1;
		x_i2 = -x_i2;
	    } 
	    if (start.y > end.y) {
		y_i1 = -y_i1;
		y_i2 = -y_i2;
	    }
	    x = start.x;
	    y = start.y;
	    for (int i = 0; i < steps; i++) {
		this->put_pixel(x, y, color.red, color.green, color.blue);
		if (D < 0) {
		    D += DE;
		    x += x_i1;
		    y += y_i1;
		} else {
		    D += DNE;
		    x += x_i2;
		    y += y_i2;
		}
	    }
	}
	void print_rectangle(const Point& top_left, const Point& bottom_right, const Color& color, bool filled) {
	    if(filled) {
		unsigned int top_x;
		unsigned int top_y;
		unsigned int bottom_x;
		unsigned int bottom_y;
		Point line_start;
		if (top_left.x > bottom_right.x) {
		    top_x = bottom_right.x;
		    top_y = bottom_right.y;
		    bottom_x = top_left.x;
		    bottom_y = top_left.y;
		    line_start = bottom_right;
		} else {
		    top_x = top_left.x;  
		    top_y = top_left.y;
		    bottom_x = bottom_right.x;
		    bottom_y = bottom_right.y;
		    line_start = top_left;
		}
		for (unsigned int y = top_y; y < bottom_y; ++y) {
		    Pixel_t* pos = this->get_pointer(line_start);
		    for (unsigned int x = top_x; x < bottom_x; ++x) {
			this->put_pixel(pos, color);
			pos++;
		    }
		    line_start.y++;
		}
	    } else {
		Point top_right(bottom_right.x, top_left.y);
		Point bottom_left(top_left.x, bottom_right.y);
		print_line(top_left, top_right, color);
		print_line(top_left, bottom_left, color);
		print_line(bottom_right, top_right, color);
		print_line(bottom_right, bottom_left, color);
	    }
	}
	void set_font(const Font& new_font) {
	    font = &new_font;
	}
	void get_pos(Point& p) {
	    p = cursor;
	}
	void set_pos(const Point& p) {
	    cursor = p;
	}
	void print_text(char* string, int len, const Color& color) {
	    unsigned char c_width = font->get_char_width();
	    unsigned char c_height = font->get_char_height();
	    Point pos;
	    get_pos(pos);
	    for(int i = 0; i < len; ++i) {
		this->blit_bitmap(pos, c_width, c_height, font->getChar(string[i]), color);
		pos.x += c_width;
		if(pos.x + c_width > this->x_max) {
		    pos.x = 0; 
		    pos.y += c_height;
		}
	    }
	    set_pos(pos);
	}
	void print_text(char* string, int len, const Color& color, const Point& pos) {
	    Point save_global_pos;
	    get_pos(save_global_pos);
	    set_pos(pos);
	    print_text(string, len, color);
	    set_pos(save_global_pos);
	}
	inline void print_sprite_alpha(const Point& p, int sprite_width, int sprite_height, const SpritePixel* sprite) {
	    Point line_start = p;
	    if (p.x + sprite_width >= this->x_max) {
		sprite_width = this->x_max;
	    }
	    if (p.y + sprite_height >= this->y_max) {
		sprite_height = this->y_max;
	    }
	    for (int y = 0; y < sprite_height; ++y) {
		Pixel_t* pos = this->get_pointer(line_start);
		for (int x = 0; x < sprite_width; ++x) {
		    Color pix_col = this->get_pixel(pos);
		    pix_col.red = pix_col.red + (((1 << (red_size - 1)) + ((int)sprite->red - (int)pix_col.red) * (int)sprite->alpha) >> red_size);
		    pix_col.green = pix_col.green + (((1 << (green_size - 1)) + ((int)sprite->green - (int)pix_col.green) * (int)sprite->alpha) >> green_size);
		    pix_col.blue = pix_col.blue + (((1 << (blue_size - 1)) + ((int)sprite->blue - (int)pix_col.blue) * (int)sprite->alpha) >> blue_size);
		    put_pixel(pos, pix_col);
		    sprite++;
		    pos++;
		}
		line_start.y += 1;
	    }
	}
};

typedef GraphicsPrinter<32, 16, 8, 0, 8, 8, 8> GraphicsPrinter_32;
typedef GraphicsPrinter<24, 16, 8, 0, 8, 8, 8> GraphicsPrinter_24;
typedef GraphicsPrinter<16, 11, 5, 0, 5, 6, 5> GraphicsPrinter_16;

#endif
