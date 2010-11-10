
#include "user/game.h"
#include "syscall/guarded_buzzer.h"
#include "user/player.h"
#include "user/monster.h"
#include "syscall/guarded_organizer.h"
#include "device/keyboard.h"

// Graphics
#include "graphics/free.c"
#include "graphics/wall.c"
#include "graphics/bricks.c"
#include "graphics/monster1.c"
#include "graphics/monster2.c"
#include "graphics/monster3.c"
#include "graphics/monster4.c"
#include "graphics/monster5.c"
#include "graphics/mango.c"
#include "graphics/player_image.c"

// Maps
#include "user/map1.h"
#include "user/map2.h"
#include "user/map3.h"


extern VESAGraphics vesa;
extern Keyboard keyboard;
extern Guarded_Organizer organizer;

Guarded_Semaphore blocker(1);

static unsigned char stack[2048];
static unsigned char stack0[2048];
static unsigned char stack1[2048];
static unsigned char stack2[2048];
static unsigned char stack3[2048];
static unsigned char stack4[2048];
static unsigned char stack5[2048];
static unsigned char stack6[2048];
static unsigned char stack7[2048];
static unsigned char stack8[2048];
static unsigned char stack9[2048];
static unsigned char stack10[2048];
static unsigned char stack11[2048];
static unsigned char stack12[2048];
static unsigned char stack13[2048];
static unsigned char stack14[2048];
static unsigned char stack15[2048];
static unsigned char stack16[2048];
static unsigned char stack17[2048];
static unsigned char stack18[2048];
static unsigned char stack19[2048];

Player player((void *)(stack+(sizeof(stack))));
Monster monster0((void *)(stack0+(sizeof(stack0))));
Monster monster1((void *)(stack1+(sizeof(stack1))));
Monster monster2((void *)(stack2+(sizeof(stack2))));
Monster monster3((void *)(stack3+(sizeof(stack3))));
Monster monster4((void *)(stack4+(sizeof(stack4))));
Monster monster5((void *)(stack5+(sizeof(stack5))));
Monster monster6((void *)(stack6+(sizeof(stack6))));
Monster monster7((void *)(stack7+(sizeof(stack7))));
Monster monster8((void *)(stack8+(sizeof(stack8))));
Monster monster9((void *)(stack9+(sizeof(stack9))));
Monster monster10((void *)(stack10+(sizeof(stack10))));
Monster monster11((void *)(stack11+(sizeof(stack11))));
Monster monster12((void *)(stack12+(sizeof(stack12))));
Monster monster13((void *)(stack13+(sizeof(stack13))));
Monster monster14((void *)(stack14+(sizeof(stack14))));
Monster monster15((void *)(stack15+(sizeof(stack15))));
Monster monster16((void *)(stack16+(sizeof(stack16))));
Monster monster17((void *)(stack17+(sizeof(stack17))));
Monster monster18((void *)(stack18+(sizeof(stack18))));
Monster monster19((void *)(stack19+(sizeof(stack19))));


Monster *monsterArray[20];

void Game::action() {

	monsterArray[0] = &monster0;
	monsterArray[1] = &monster1;
	monsterArray[2] = &monster2;
	monsterArray[3] = &monster3;
	monsterArray[4] = &monster4;
	monsterArray[5] = &monster5;
	monsterArray[6] = &monster6;
	monsterArray[7] = &monster7;
	monsterArray[8] = &monster8;
	monsterArray[9] = &monster9;
	monsterArray[10] = &monster10;
	monsterArray[11] = &monster11;
	monsterArray[12] = &monster12;
	monsterArray[13] = &monster13;
	monsterArray[14] = &monster14;
	monsterArray[15] = &monster15;
	monsterArray[16] = &monster16;
	monsterArray[17] = &monster17;
	monsterArray[18] = &monster18;
	monsterArray[19] = &monster19;

	
	Guarded_Buzzer buzzer;

	vesa.print_text("Press any Key to start the game", 31, Color(0xff, 0xff, 0xff));

	vesa.switch_buffers();
	vesa.scanout_frontbuffer();

	keyboard.getkey();
	vesa.clear_screen();
	vesa.switch_buffers();
	vesa.clear_screen();


	init(level);

	organizer.ready(player);
	organizer.ready(monster0);
	organizer.ready(monster1);
	organizer.ready(monster2);
	organizer.ready(monster3);
	organizer.ready(monster4);
	organizer.ready(monster5);
	organizer.ready(monster6);
	organizer.ready(monster7);
	organizer.ready(monster8);
	organizer.ready(monster9);
	organizer.ready(monster10);
	organizer.ready(monster11);
	organizer.ready(monster12);
	organizer.ready(monster13);
	organizer.ready(monster14);
	organizer.ready(monster15);
	organizer.ready(monster16);
	organizer.ready(monster17);
	organizer.ready(monster18);
	organizer.ready(monster19);
	
	drawMap();
	running = true;

	while(true) {
		while(running){
			drawMap();
		}
		
		buzzer.set(222);
		buzzer.sleep();
		gameOverString = "";
		gameOverStringLen = 0;
		game.block();
		fruitLeft = fruitAmt;
		init(level);
		game.unblock();
		drawMap();
		running = true;
	}
}

void Game::init(int mapIndex) {
	initMap(mapIndex);
}

void Game::initMap(int mapIndex) {
	int curMonster = 0;
	fruitAmt = 0;

	for (int i = 0; i < MAPSIZE; i++) {
		for (int j = 0; j < MAPSIZE; j++) {

			if (mapIndex == 0) {
				map[j][i] = map1[i][j];
			} else if (mapIndex == 1) {
				map[j][i] = map2[i][j];
			} else if (mapIndex == 2) {
				map[j][i] = map3[i][j];
			}


			if (map[j][i] == PLAYER) {
				player.init(j, i);
			} else if (map[j][i] == MONSTER) {
				monsterArray[curMonster++]->init(j, i, 0);
			} else if (map[j][i] == FRUIT) {
				fruitAmt++;
			}
			
		}
	}

	fruitLeft = fruitAmt;
}

void Game::drawMap() {
	vesa.clear_screen();
	vesa.switch_buffers();
	vesa.clear_screen();
	for (int i = 0; i < MAPSIZE; i++) {
		for (int j = 0; j < MAPSIZE; j++) {
			Point topLeft(i*FIELDSIZE, j*FIELDSIZE);
			switch (map[i][j]) {
				case FREE:
						//vesa.print_rectangle(Point(i*FIELDSIZE, j*FIELDSIZE), Point(i*FIELDSIZE+FIELDSIZE, j*FIELDSIZE+FIELDSIZE), Color(0,0,0xff), true);
						vesa.print_sprite_alpha(topLeft, FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(free_image.pixel_data));
						break;
				case WALL:
						//vesa.print_rectangle(Point(i*FIELDSIZE, j*FIELDSIZE), Point(i*FIELDSIZE+FIELDSIZE, j*FIELDSIZE+FIELDSIZE), Color(0xdd,0,0xff), true);
						vesa.print_sprite_alpha(topLeft, FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(wall_image.pixel_data));
						break;
				case BORDER:
						//vesa.print_rectangle(Point(i*FIELDSIZE, j*FIELDSIZE), Point(i*FIELDSIZE+FIELDSIZE, j*FIELDSIZE+FIELDSIZE), Color(0,0xff,0xff), true);
						vesa.print_sprite_alpha(topLeft, FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(bricks_image.pixel_data));
						break;
				case PLAYER:
						//vesa.print_rectangle(Point(i*FIELDSIZE, j*FIELDSIZE), Point(i*FIELDSIZE+FIELDSIZE, j*FIELDSIZE+FIELDSIZE), Color(12,0,0), true);
						vesa.print_sprite_alpha(topLeft, FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(player_image.pixel_data));
						break;
				case MONSTER:
						//vesa.print_rectangle(Point(i*FIELDSIZE, j*FIELDSIZE), Point(i*FIELDSIZE+FIELDSIZE, j*FIELDSIZE+FIELDSIZE), Color(0x22,0x11,0xff), true);

						vesa.print_sprite_alpha(topLeft, FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(monster2_image.pixel_data));
						break;
				case FRUIT:
						//vesa.print_rectangle(Point(i*FIELDSIZE, j*FIELDSIZE), Point(i*FIELDSIZE+FIELDSIZE, j*FIELDSIZE+FIELDSIZE), Color(0,0,0xff), true);
						vesa.print_sprite_alpha(topLeft, FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(mango_image.pixel_data));
						break;
			}
		}
	}


	for (int i = 0; i < MAPSIZE; i++) {
		for (int j = 0; j < (1024-768)/24; j++) {
			vesa.print_sprite_alpha(Point(768 + j*24, i*24), FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(bricks_image.pixel_data));
		}
	}

	vesa.print_text("FRUITY BENJADIRK", 16, Color(0, 0, 0), Point(280, 2));
	vesa.print_text("SCORE", 5, Color(0, 0, 0), Point(768, 50));	

	int yoffset = 0;


	int j = 0;
	for (int i = 0; i < fruitAmt - fruitLeft; i++) {
			if (i != 0 &&i % 5 == 0) yoffset++;
		vesa.print_sprite_alpha(Point(850 + j*24, 50 + yoffset*24), FIELDSIZE, FIELDSIZE, reinterpret_cast<const SpritePixel*>(mango_image.pixel_data));
		j++;
		j = j%5;
	}
	for (int i = fruitAmt - fruitLeft; i < fruitAmt; i++) {
			if (i != 0 && i % 5 == 0) yoffset++;
		vesa.print_rectangle(Point(850 + j*24, 50 + yoffset*24), Point(850 + j*24 + 24, 50+24 + yoffset*24), Color(0,0,0), true);
		j++;
		j = j%5;
	}


	if (gameOverStringLen != 0) vesa.print_rectangle(Point(235, 350), Point(515, 390), Color(0x0, 0x0, 0x0), true);
	vesa.print_text(gameOverString, gameOverStringLen, Color(0xff, 0xff, 0xff), Point(245, 360));

	vesa.switch_buffers();
	vesa.scanout_frontbuffer();
}

Point Game::getPlayerPosition() {
	return player.getPosition();
}

bool Game::isRunning() {
	return running;
}

void Game::gameOver(bool won) {
	score = 0;
	if (won) {
		if (level >= 3) {
			gameOverString = "YOU DID IT!!!!!!!!!!!!";
			gameOverStringLen = 22;
		} else {
			gameOverString = "YOU WIN! NEXT LEVEL!!!";
			gameOverStringLen = 22;
			level++;
		}
	}
	else {
		gameOverString = "YOU LOOOSE!!! RETRY!!!";
		gameOverStringLen = 22;
	}
	running = false;
}

void Game::block() {
	blocker.wait();
}

void Game::unblock() {
	blocker.signal();
}

bool Game::increaseScore() {
	bool over = false;
	fruitLeft--;
	if (fruitLeft == 0) {
		//GEWONNEN!
		over = true;
	}
	else {
		score+=10;
	}
	return over;
}

Game::Game(void *tos) : Thread(tos) { 
		monsterTimeToWait = 20;
		score = 0;
		fruitAmt = 5;
		fruitLeft = fruitAmt;
		gameOverStringLen = 0;
		gameOverString = "";
		level = 0;
	}

void Game::restartLevel() {
	game.block();
	init(level);
	game.unblock();
}

void Game::resetGame() {
	//TODO
}
