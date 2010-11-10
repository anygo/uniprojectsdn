/*#include "user/santas_house.h"

#include "syscall/guarded_buzzer.h"

extern VESAGraphics vesa;

#include "graphics/sun.c"
#include "graphics/cat.c"

void Santas_House::action() {
    while(1) {
	Point pos(100, 200);
	for(int i = 0; i < 200; i+=5) {
	    pos.x += 5;
	    pos.y += 5;
	    print_house(pos);
	}
	for(int i = 0; i < 200; i+=5) {
	    pos.x += 5;
	    pos.y -= 5;
	    print_house(pos);
	} 
	for(int i = 0; i < 400; i+=5) {
	    pos.x -= 5;
	    print_house(pos);
	} 
    } 
}

void Santas_House::print_house(const Point& p) {
    Guarded_Buzzer waiter;
    Point left_up;
    Point left_down;
    Point right_up;
    Point right_down;
    Point spire;
    Point text;
    Point sunpos;
	Point ursprung;
    waiter.set(30);
    left_up = p; 
    right_down = Point(p.x + 200, p.y + 200);
    left_down = Point(left_up.x, right_down.y);
    right_up = Point(right_down.x, left_up.y);
    spire = Point(p.x+100, p.y-100);
    text = Point(p.x, p.y-130);
    sunpos = Point(right_up.x - 60, spire.y - 80);

    vesa.clear_screen();
	
//	vesa.print_sprite_alpha(Point(0, 0), 32, 32, reinterpret_cast<const SpritePixel*>(gimp_image.pixel_data));
    
	vesa.print_text("Haus vom Nikolaus", 17, Color(0x90, 0x0f, 0xd0), text);
    vesa.print_rectangle(left_up, right_down, Color(0xff, 0, 0), true);
    vesa.print_rectangle(left_up, right_down, Color(0, 0, 0xff), false);
    vesa.print_line(left_up, right_down, Color(0, 0, 0xff));
    vesa.print_line(left_down, right_up, Color(0, 0, 0xff));
    vesa.print_line(left_up, spire, Color(0, 0, 0xff));
    vesa.print_line(right_up, spire, Color(0, 0, 0xff));
    vesa.print_sprite_alpha(sunpos, 160, 160, reinterpret_cast<const SpritePixel*>(sun.pixel_data));
    vesa.print_sprite_alpha(Point(right_down.x + 3, right_down.y-80), 320, 240, reinterpret_cast<const SpritePixel*>(cat.pixel_data));
    vesa.switch_buffers();
    waiter.sleep(); 
}

*/
