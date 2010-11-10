#ifndef __GRAPHICSTYPES_H__
#define __GRAPHICSTYPES_H__

struct Point {
    unsigned int x;
    unsigned int y;
    Point() : x(0), y(0) { }
    Point(unsigned int x, unsigned int y) : x(x), y(y) { } 
};

struct Color {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    Color(unsigned char red, unsigned char green, unsigned char blue) : red(red), green(green), blue(blue) { }
    Color() {}
};

struct SpritePixel {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha; 
    SpritePixel(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha) : red(red), green(green), blue(blue), alpha(alpha) { }
    SpritePixel() {}
};

struct Triangle {
    Point points[3];
    Triangle(Point& p1, Point& p2, Point& p3) {
	points[0] = p1;
	points[1] = p2;
	points[2] = p3;
    }
};

#endif
