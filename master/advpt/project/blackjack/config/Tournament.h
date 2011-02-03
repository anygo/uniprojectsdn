//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CONFIG_TOURNAMENT_H_
#define _BLACKJACK_CONFIG_TOURNAMENT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blackjack/util/Types.h>


namespace blackjack {

//*************************************************************************************************
/*!\brief Specifies the number of games in every qualification playoff.
//
// This value specifies the number of games played in every qualification playoff of the
// black jack tournament.
*/
const size_t gamesPerQualification( 10 );
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Specifies the maximum number of players in every game.
//
// This value specifies the maximum number of players per game.
*/
const size_t playersPerGame( 6 );
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Specifies the round played in every game.
//
// This value specifies the number of rounds in each game of black jack, i.e., the number of
// times cards are dealt.
*/
const size_t roundsPerGame( 100000 );
//*************************************************************************************************

} // namespace blackjack

#endif
