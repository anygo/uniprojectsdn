
// NICHT DIE DEFINES VERGESSEN!!!!!!

#ifndef __MAP3__
#define __MAP3__

/*
 * 0: FREE
 * 1: WALL
 * 2: BORDER
 * 3: PLAYER  (bitte GENAU 1x PLAYER!!!)
 * 4: MONSTER (aktuell MAXIMAL 10 MONSTER)
 * 5: FRUIT
 *
 */ 

int map3[32][32] = {
{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
{2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4,5,0,0,0,0,0,0,0,0,0,0,0,0,2,5,2},
{2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,0,4,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,2,2,2,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,2,2,2,2,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,5,2},
{2,0,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,2,2,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,2},
{2,0,2,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,2,2,2,2,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,2,2,2,0,2},
{2,0,2,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,2,0,2,0,2,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2},
{2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,4,5,0,0,0,0,0,0,0,0,0,0,0,0,2,5,2},
{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},

};

#endif
