#include "user/player.h"

void Player::action() {
	while (true) {
		setPosition(start.x, start.y);
	
		do {
			Key k = keyboard.getkey();
			game.block();
			int over = 0;
			switch (k.scancode()) {
				case 72:
						over = move(Game::UP);
						break;
				case 80:
						over = move(Game::DOWN);
						break;
				case 75:
						over = move(Game::LEFT);
						break;
				case 77:
						over  = move(Game::RIGHT);
						break;
				default:
						if (k.ascii() == 'p') {
							keyboard.getkey();
						}
			}
			game.unblock();
			if (over == 1) {
				game.gameOver(false);
				game.block();
				game.unblock();
			}
			else if (over == 2) {
				game.gameOver(true);
				game.block();
				game.unblock();
			}
		} while (game.isRunning());
	}
}

void Player::setPosition(int x, int y) {
	game.map[position.x][position.y] = FREE;
	game.map[x][y] = PLAYER;
	position.x = x;
	position.y = y;
}

int Player::move(int direction) {
	int over = 0;
	Point p = position;
	switch (direction) {
			case Game::DOWN:
				p.y++;
				break;
			case Game::UP:
				p.y--;
				break;
			case Game::RIGHT:
				p.x++;
				break;
			case Game::LEFT:
				p.x--;
				break;
	}
	

	if (game.map[p.x][p.y] == MONSTER) {
		over = 1;
	}
	else if (game.map[p.x][p.y] == FREE || game.map[p.x][p.y] == WALL) {
		setPosition(p.x, p.y);
	}
	else if (game.map[p.x][p.y] == FRUIT) {
		setPosition(p.x, p.y);
		bool tmp = game.increaseScore();
		if (tmp) over = 2;
	}
	return over;
}
