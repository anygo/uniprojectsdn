#include "user/monster.h"
#include "syscall/guarded_buzzer.h"

void Monster::action() {
	Guarded_Buzzer waiter;
	bool over;

	while(true) {
//		game.block();
//		cur.x = start.x;
//		cur.y = start.y;
//		setPosition(start.x, start.y);
//		game.unblock();
		do {
			waiter.set(game.getMonsterTimeToWait());
			if (active()) {
				game.block();
				over = move();
				game.unblock();
				if (over) {
						game.gameOver(false);
				} else {
					waiter.sleep();
				}
			} 
			else {
				waiter.sleep();
			}
		} while (game.isRunning());
	}
}

void Monster::setPosition(int x, int y) {
	game.map[cur.x][cur.y] = previous;
	previous = game.map[x][y];
	game.map[x][y] = MONSTER;
	cur.x = x;
	cur.y = y;

	//game.drawMap();
}

bool Monster::move() {
	bool over = false;
	Random TESTrandom(cur.x);
	Point p(cur.x, cur.y);
	Point playerPos = game.getPlayerPosition();
	
	if(game.map[p.x][p.y+1] == PLAYER ||
		game.map[p.x][p.y-1] == PLAYER ||
		game.map[p.x+1][p.y] == PLAYER ||
		game.map[p.x-1][p.y] == PLAYER) {
			over = true;
	}

	if(strategy == DIRECT) {
		int distanceX = playerPos.x - p.x;
		int distanceY = playerPos.y - p.y;

		if(distanceX <= 0 && distanceY <= 0) {
			if(distanceX <= distanceY && game.map[p.x-1][p.y] == FREE) {
				p.x--;
			} else if(distanceX >= distanceY && game.map[p.x][p.y-1] == FREE) {
				p.y--;
			} else if(game.map[p.x-1][p.y] == FREE) {
				p.x--;
			} else if(game.map[p.x][p.y-1] == FREE) {
				p.y--;
			} else if(game.map[p.x+1][p.y] == FREE) {
				p.x++;
			} else if(game.map[p.x][p.y+1] == FREE) {
				p.y++;
			}
		} else if(distanceX <= 0 && distanceY >= 0) {
			if(abs(distanceX) >= abs(distanceY) && game.map[p.x-1][p.y] == FREE) {
				p.x--;
			} else if(abs(distanceX) <= abs(distanceY) && game.map[p.x][p.y+1] == FREE) {
				p.y++;
			} else if(game.map[p.x-1][p.y] == FREE) {
				p.x--;
			} else if(game.map[p.x][p.y+1] == FREE) {
				p.y++;
			} else if(game.map[p.x+1][p.y] == FREE) {
				p.x++;
			} else if(game.map[p.x][p.y-1] == FREE) {
				p.y--;
			}
		} else if(distanceX >= 0 && distanceY <= 0) {
			if(abs(distanceX) >= abs(distanceY) && game.map[p.x+1][p.y] == FREE) {
				p.x++;
			} else if(abs(distanceX) <= abs(distanceY) && game.map[p.x][p.y-1] == FREE) {
				p.y--;
			} else if(game.map[p.x+1][p.y] == FREE) {
				p.x++;
			} else if(game.map[p.x][p.y-1] == FREE) {
				p.y--;
			} else if(game.map[p.x-1][p.y] == FREE) {
				p.x--;
			} else if(game.map[p.x][p.y+1] == FREE) {
				p.y++;
			}
		} else if(distanceX >= 0 && distanceY >= 0) {
			if(distanceX >= distanceY && game.map[p.x+1][p.y] == FREE) {
				p.x++;
			} else if(distanceX <= distanceY && game.map[p.x][p.y+1] == FREE) {
				p.y++;
			} else if(game.map[p.x+1][p.y] == FREE) {
				p.x++;
			} else if(game.map[p.x][p.y+1] == FREE) {
				p.y++;
			} else if(game.map[p.x-1][p.y] == FREE) {
				p.x--;
			} else if(game.map[p.x][p.y-1] == FREE) {
				p.y--;
			}
		}
	} else if(strategy == RANDOM) {
		int posibleDirections = 0;
		if(game.map[p.x-1][p.y] == FREE)
			posibleDirections++;
		if(game.map[p.x+1][p.y] == FREE)
			posibleDirections++;
		if(game.map[p.x][p.y-1] == FREE)
			posibleDirections++;
		if(game.map[p.x][p.y+1] == FREE)
			posibleDirections++;
		if(posibleDirections > 0) {
			int posChange = TESTrandom.number() % 4;
			while(true) {
				if(posChange == 0) {
					if(game.map[p.x][p.y+1] == FREE) {
						p.y++;
						break;
					}
					else {
						posChange++;
					}
				} else if(posChange == 1) {
					if(game.map[p.x][p.y-1] == FREE) {
						p.y--;
						break;
					}
					else {
						posChange++;
					}
				} else if(posChange == 2) {
					if(game.map[p.x+1][p.y] == FREE) {
						p.x++;
						break;
					}
					else {
						posChange++;
					}
				} else if(posChange == 3) {
					if(game.map[p.x-1][p.y] == FREE) {
						p.x--;
						break;
					}
					else {
						posChange = 0;
					}
				}
			}
		}
	}
	
	if (game.map[p.x][p.y] == FREE) {	
		setPosition(p.x, p.y);
	}
	return over;
}

