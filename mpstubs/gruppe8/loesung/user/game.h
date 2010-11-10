#ifndef __GAME__
#define __GAME__

#include "syscall/thread.h"
#include "syscall/guarded_vesagraphics.h"
#include "library/random.h"
#include "syscall/guarded_semaphore.h"


// Defines
#define MAPSIZE 32
#define FIELDSIZE (768/MAPSIZE)
#define SCREENFIELDS (1024/MAPSIZE)


enum feld {
	FREE = 0, WALL = 1, BORDER = 2, PLAYER = 3, MONSTER = 4, FRUIT = 5
};

enum strategy {
	DIRECT = 0, RANDOM = 1
};

typedef struct Position Position;

class Game : public Thread {
private:
	void initMap(int mapIndex);
	bool running;
	int monsterTimeToWait;
	int score;
	int fruitAmt;
	int gameOverStringLen;
	char *gameOverString;
	int level;

public:	
	int map[MAPSIZE][MAPSIZE];
	int fruitLeft;
	enum posChange {
		UP, DOWN, LEFT, RIGHT
	};

	Game(void *tos);
	virtual void action();
	void drawMap();
	void updateScreen();
	void gameOver(bool won);
	bool isRunning();
	Point getPlayerPosition();
	void block();
	void unblock();
	void setMonsterTimeToWait(int ticks) { monsterTimeToWait = ticks; }
	int getMonsterTimeToWait() { return monsterTimeToWait; }
	bool increaseScore(); 
	void init(int mapIndex);
	void restartLevel();
	void resetGame();
};

#endif
