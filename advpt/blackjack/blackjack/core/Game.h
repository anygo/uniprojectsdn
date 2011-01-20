//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_GAME_H_
#define _BLACKJACK_CORE_GAME_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <string>
#include <vector>
#include <blackjack/core/CardDeck.h>
#include <blackjack/core/CardHand.h>
#include <blackjack/core/Player.h>
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
/*!\brief Implementation of the rules for a black jack game.
//
// The game class represents a black jack game with several rounds and a card deck consisting
// of 6*52 cards. Players are added to a game via the addPlayer() function in the order in
// which they will receive cards, i.e., the first player added to a game will be the first to
// get a card in each round, the second player will be the second, etc. All player joining a
// game will be introduced to each other via the introduce function.
*/
class Game : private NonCopyable
{
 private:
   //**Nested class definition*********************************************************************
   /*!\brief Control wrapper for a black jack player participating in a game.
   //
   // The ControlledPlayer class is a wrapper class for a player participating the a game of
   // black jack. This class works as an observer and guarantees the correct adherence to the
   // rules of the game.
   */
   struct ControlledPlayer
   {
    public:
      explicit ControlledPlayer( PlayerID player );

      const std::string& name    () const { return player_->name(); }
      bool               offer   ();
      void               give    ( CardID card );
      void               introduce      ( PlayerID player );
      void               playerDrawsCard( const ControlledPlayer& player, CardID card );
      void               playerStands   ( const ControlledPlayer& player );
      void               newRound( bool shuffle );
      void               newGame ();

      bool               isOut() const { return out_; }
      value_t            count() const { return value_; }

      size_t wins     () const { return player_->wins(); }
      void   addWin   ()       { player_->addWin(); }
      void   resetWins()       { player_->resetWins(); }

      bool out_;
      PlayerID player_;
      value_t value_;
      CardHand hand_;
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   typedef std::vector<ControlledPlayer>  Players;        // Container type for the participating players.   typedef Players::iterator             Iterator;       // Iterator over non-constant players.
   typedef Players::iterator              Iterator;       // Iterator over non-constant players.
   typedef Players::const_iterator        ConstIterator;  // Iterator over constant players.
   //**********************************************************************************************

 public:
   //**Constructors and destructor*****************************************************************
   Game();
   template< typename Iter > Game( Iter first, Iter last );
   ~Game();
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   size_t size() const { return players_.size(); }
   void   addPlayer( PlayerID player );
   void   play( size_t games );
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   CardDeck deck_;     // Card deck used in the game.
   Players  players_;  // The players participating in the game.
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for a game of black jack.
//
// \param first Iterator to the first player entering the black jack game.
// \param last Iterator one past the last player entering the black jack game.
*/
template< typename Iter >
Game::Game( Iter first, Iter last )
   : deck_ ()    // Card deck used in the game
   , players_()  // The players participating in the game
{
   for( ; first!=last; ++first )
      addPlayer( *first );
}
//*************************************************************************************************

} // namespace blackjack

#endif
