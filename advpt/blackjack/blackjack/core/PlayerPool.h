//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_PLAYERPOOL_H_
#define _BLACKJACK_CORE_PLAYERPOOL_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <functional>
#include <vector>
#include <blackjack/core/Types.h>
#include <blackjack/util/NonCopyable.h>
#include <blackjack/util/Types.h>


namespace blackjack {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Pool of all players participating in the black jack tournament.
//
// This class represents the pool of all players participating in the black jack tournament.
// All players deriving from the RegisteredPlayer class are automatically register in this
// player pool.
*/
class PlayerPool : private NonCopyable
{
 private:
   //**Nested functor Compare**********************************************************************
   struct Compare : public std::binary_function<PlayerID,PlayerID,bool>
   {
      //**Binary function call operator************************************************************
      inline bool operator()( PlayerID player1, PlayerID player2 ) const {
         return player1->wins() > player2->wins();
      }
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   typedef std::vector<Player*>     Players;        // Container for all players.
   typedef Players::iterator        Iterator;       // Iterator over constant players.
   typedef Players::const_iterator  ConstIterator;  // Iterator over non-constant players.
   //**********************************************************************************************

   //**Constructor/destructor**********************************************************************
   // No excplicitly declared constructor.
   // No excplicitly declared destructor.
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   static inline size_t   size ();
   static inline Iterator begin();
   static inline Iterator end  ();

   static inline void     pushBack( PlayerID player );

   static inline Iterator erase( Iterator pos );
   static inline Iterator erase( Iterator first, Iterator last );

   static        void     shuffle();
   static        void     sort   ();

   static inline Players& thePlayers();
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename T > friend class RegisteredPlayer;
   friend class Tournament;
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the total size of the player pool
//
// \return The total size of the player pool.
*/
inline size_t PlayerPool::size()
{
   return thePlayers().size();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first player contained in the player pool.
//
// \return Iterator to the first player contained in the player pool.
*/
inline PlayerPool::Iterator PlayerPool::begin()
{
   return thePlayers().begin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator one past the last player contained in the player pool.
//
// \return Iterator one past the last player contained in the player pool.
*/
inline PlayerPool::Iterator PlayerPool::end()
{
   return thePlayers().end();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Adding a new player to the player pool.
//
// \param player The new player to be added to the player pool.
// \return void
*/
inline void PlayerPool::pushBack( PlayerID player )
{
   thePlayers().push_back( player );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing a single player from the player pool.
//
// \param pos Iterator to the player to be erased from the player pool.
// \return Iterator to the player one past the erased player.
*/
inline PlayerPool::Iterator PlayerPool::erase( Iterator pos )
{
   Players& players( thePlayers() );
   return players.erase( pos );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing a group of players from the player pool.
//
// \param first Iterator to the first player to be erased from the player pool.
// \param last Iterator one past the last player to be erased from the player pool.
// \return Iterator to the player one past the last erased player.
*/
inline PlayerPool::Iterator PlayerPool::erase( Iterator first, Iterator last )
{
   Players& players( thePlayers() );
   return players.erase( first, last );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the players contained in the player pool.
//
// \return Vector of all players contained in the player pool.
*/
inline PlayerPool::Players& PlayerPool::thePlayers()
{
   static Players players;
   return players;
}
//*************************************************************************************************

} // namespace blackjack

#endif
