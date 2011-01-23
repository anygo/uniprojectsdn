//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <cstdlib>
#include <blackjack/core/Player.h>
#include <blackjack/core/PlayerPool.h>


namespace blackjack {

//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Shuffling the player pool.
//
// \return void.
*/
void PlayerPool::shuffle()
{
   Players& players( thePlayers() );
   const int size ( static_cast<int>( players.size() ) );
   const int count( 10*size );

   for( int i=0; i<count; ++i ) {
      std::swap( players[rand()%size], players[rand()%size] );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sorting the player pool in descending order of wins.
//
// \return void.
*/
void PlayerPool::sort()
{
   using std::sort;
   Players& players( thePlayers() );
   std::sort( players.begin(), players.end(), Compare() );
}
//*************************************************************************************************

} // blackjack
