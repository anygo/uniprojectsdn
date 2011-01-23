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

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <blackjack/config/Tournament.h>
#include <blackjack/core/Game.h>
#include <blackjack/core/PlayerPool.h>
#include <blackjack/core/Tournament.h>
#include <blackjack/util/ScopedArray.h>


namespace blackjack {

//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Running the black jack tournament.
//
// This function implements the organization of the black jack tournament.
*/
void Tournament::run()
{
   PlayerPool playerpool;
   size_t size( playerpool.size()  );
   char c;

   // Early exit in case less than two players want to participate
   if( size < 2 ) {
      std::cerr << "\n Not enough players to run a blackjack tournament\n\n";
      return;
   }

   // Setting the random number generator
   std::srand( std::time( 0 ) );

   // Configuring the standard input stream
   std::cin.unsetf( std::istream::skipws );

   // Playing the qualification playoffs
   size_t qualification( 0 );

   while( size > playersPerGame )
   {
      ++qualification;

      PlayerPool::Iterator begin( playerpool.begin() );
      PlayerPool::Iterator end  ( playerpool.end()   );

      for( PlayerPool::Iterator player=begin; player!=end; ++player ) {
         (*player)->resetWins();
      }

      std::cout << "\n Starting QUALIFICATION PLAYOFF " << qualification << ":\n";
      for( PlayerPool::Iterator player=begin; player!=end; ++player )
         std::cout << "   " << (*player)->name() << "\n";
      std::cout << std::endl;

      const size_t numGames( size/playersPerGame+1 );

      // Pausing the tournament
      std::cout << " Press a key to continue...";
      std::cin >> c;
      std::cout << "\n";

      // Playing the specified number of games in each qualification
      for( size_t i=0; i<gamesPerQualification; ++i )
      {
         ScopedArray<Game> games( new Game[numGames] );
         Game* gbegin( games.get() );
         Game* gend  ( games.get()+numGames );

         playerpool.shuffle();

         {
            PlayerPool::Iterator player( begin );
            Game* game( gbegin );

            while( player != end ) {
               game->addPlayer( *player );
               ++player;
               ++game;
               if( game == gend ) game = gbegin;
            }
         }

         for( Game* game=gbegin; game!=gend; ++game ) {
            game->play( roundsPerGame );
         }
      }

      // Sorting the players according to their wins
      playerpool.sort();

      std::cout << " Results:\n";
      for( PlayerPool::Iterator player=begin; player!=end; ++player )
         std::cout << "   " << (*player)->name() << ": wins=" << (*player)->wins() << "\n";
      std::cout << "\n" << std::endl;

      // Eliminating the weakest players
      size /= 2;
      if( size < playersPerGame )
         size = playersPerGame;

      PlayerPool::Iterator last ( begin+size-1 );  // Last player to enter the next qualification
      PlayerPool::Iterator first( begin+size   );  // First player to drop out

      while( (*first)->wins() == (*last)->wins() && first != end )
         ++first;

      playerpool.erase( first, end );
   }

   // Preparing the final
   std::cout << "\n Starting the FINAL:\n";

   playerpool.shuffle();

   PlayerPool::Iterator begin( playerpool.begin() );
   PlayerPool::Iterator end  ( playerpool.end()   );

   for( PlayerPool::Iterator player=begin; player!=end; ++player )
      std::cout << "   " << (*player)->name() << "\n";
   std::cout << std::endl;

   for( PlayerPool::Iterator player=begin; player!=end; ++player ) {
      (*player)->resetWins();
   }

   // Pausing the tournament
   std::cout << " Press a key to continue...";
   std::cin >> c;
   std::cout << "\n";

   // Playing the final
   Game game( begin, end );
   game.play( roundsPerGame );

   // Sorting the players according to their wins
   playerpool.sort();

   std::cout << " Results:\n";
   for( PlayerPool::Iterator player=begin; player!=end; ++player )
      std::cout << "   " << (*player)->name() << ": wins=" << (*player)->wins() << "\n";
   std::cout << "\n" << std::endl;

   // Resetting the standard input stream
   std::cin.setf( std::istream::skipws );
}
//*************************************************************************************************

} // namespace blackjack
