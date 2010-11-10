#ifndef __MONSTER__
#define __MONSTER__

#include "syscall/thread.h"
#include "user/game.h"


extern Game game;

class Monster : public Thread {
private:
	int strategy;
	int previous;
	Point start;
	Point cur;
	int timeToWait;
	bool monsterIsActive;
	

public:
	Monster(void *tos) : Thread(tos) { 
		deactivate();
	}
	virtual void action();
	bool move();
	void setPosition(int x, int y);
	void init(int startx, int starty, int strat = 0) {
		start.x = cur.x = startx;
		start.y = cur.y = starty;
		previous = FREE;
		strategy = strat;
		activate();
	}
	bool active() { return monsterIsActive; }
	void activate() { monsterIsActive = true; }
	void deactivate() { monsterIsActive = false; }
};

#endif
