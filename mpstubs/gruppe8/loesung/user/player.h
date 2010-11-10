#ifndef __PLAYER__
#define __PLAYER__

#include "syscall/thread.h"
#include "user/game.h"
#include "device/keyboard.h"

extern Game game;
extern Keyboard keyboard;

class Player : public Thread {
private:
	Point position;
	Point start;

public:
	Player(void *tos) : Thread(tos) { 
	}
	virtual void action();
	void setPosition(int x, int y);
	Point getPosition() { return position; }
	int move(int direction);
	void init(int startx, int starty) {
		start.x = position.x = startx;
		start.y = position.y = starty;
		setPosition(startx, starty);	
	};
};

#endif
