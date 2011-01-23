//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_PLAYER_H_
#define _BLACKJACK_CORE_PLAYER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <string>
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
/*!\brief Impementation of black jack player.
//
// The Player class represent the base class for all black jack players. Each player of
// the black jack tournament indirectly inherits from this base class by deriving from the
// RegisteredPlayer class. The Player class defines the interface that can be used to
// implement a suitable black jack strategy in order to win the tournament. The following
// gives a short introduction to the purpose of every function that might be overloaded:
//
//  - introduce: Before every new game, all players are introduced to each other. This
//               is done by using the introduce function, which can be used to remember
//               the other players participating in the game.
//  - playerDrawCard: This function notifies the player that another player has drawn
//                    a specific card. Via this function it is for example possible to
//                    remember which player holds which cards in his/her hands.
//  - playerStands: This function notifies the player that another player has decided
//                  to stand, i.e., not to take another card. This function can be used
//                  to optimize the card drawing strategy.
//  - offer: This function requests whether the player wishes to draw another card. The
//           player has to reply by either returning true if he want to draw another card
//           or by returning false if he chooses to stand. This function has to be implemented
//           in order to be able to instantiate the player.
//  - give: The give function is used to give the player a new card into his/her hand. This
//          function has to be implemented in order to be able to instantiate the player.
//  - newRound: This function signals the beginning of a new round, i.e., new cards are
//              drawn. In case the shuffle parameter is true, the deck has been shuffled
//              prior to this new round, otherwise no re-shuffling has taken place.
//  - newGame: This function signals the beginning of a new game, i.e., new involved
//             players and a new card deck.
*/
class Player : private NonCopyable
{
 public:
   //**Constructor*********************************************************************************
   Player( const std::string& fullname );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   virtual ~Player();
   //**********************************************************************************************

 public:
   //**Utility functions***************************************************************************
   inline const std::string& name() const { return name_; }
   inline size_t             wins() const { return wins_; }
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   void addWin   () { ++wins_; }
   void resetWins() { wins_ = 0u; }

   virtual void introduce      ( PlayerID player );
   virtual void playerDrawsCard( PlayerID player, CardID card );
   virtual void playerStands   ( PlayerID player );

   virtual bool offer()               = 0;
   virtual void give ( CardID card  ) = 0;
   virtual void newRound( bool shuffle );
   virtual void newGame ();
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const std::string name_;  // Name of the player
   size_t            wins_;  // Accumulated number of wins
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   friend class Game;
   friend class Tournament;
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
