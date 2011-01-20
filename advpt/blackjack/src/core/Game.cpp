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

#include <blackjack/config/Tournament.h>
#include <blackjack/core/Game.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for the Game class.
*/
Game::Game()
   : deck_ ()    // Card deck used in the game
   , players_()  // The players participating in the game
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the Game class.
*/
Game::~Game()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Adding a player to the game.
//
// \param player The new player to be added to the game.
// \return void
*/
void Game::addPlayer( PlayerID player )
{
   player->newGame();

   for( Iterator other=players_.begin(); other!=players_.end(); ++other ) {
      other->introduce ( player );
      player->introduce( other->player_ );
   }

   players_.push_back( ControlledPlayer( player ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Playing several rounds of black jack.
//
// \param rounds The number of rounds to be played.
// \return void
*/
void Game::play( size_t rounds )
{
   // Early exit in case no players or only a single player participate in the game
   if( players_.size() < 2 ) return;

   const size_t num( players_.size() );

   CardID card;
   size_t out( 0u );
   size_t maxvalue( 0u );

   for( size_t round=0u; round<rounds; ++round )
   {
      const Iterator begin( players_.begin() );
      const Iterator end  ( players_.end()   );

      // Drawing the first card
      for( Iterator player=begin; player!=end; ++player ) {
         card = deck_.draw();
         player->give( card );
         for( Iterator other=begin; other!=end; ++other ) {
            if( player != other )
               other->playerDrawsCard( *player, card );
         }
      }

      // Drawing the second card
      for( Iterator player=begin; player!=end; ++player ) {
         card = deck_.draw();
         player->give( card );
         for( Iterator other=begin; other!=end; ++other ) {
            if( player != other )
               other->playerDrawsCard( *player, card );
         }
      }

      // Drawing all other cards
      while( out < num )
      {
         for( Iterator player=begin; player!=end; ++player )
         {
            if( player->isOut() ) {
               continue;
            }

            if( !player->offer() ) {
               for( Iterator other=begin; other!=end; ++other ) {
                  if( player != other )
                     other->playerStands( *player );
               }
               ++out;
               continue;
            }

            card = deck_.draw();
            player->give( card );

            for( Iterator other=begin; other!=end; ++other ) {
               if( player != other )
                  other->playerDrawsCard( *player, card );
            }

            if( player->count() > 21u ) {
               ++out;
            }
         }
      }

      // Resetting the out counter
      out = 0u;

      // Estimating the winners
      maxvalue = 0u;
      for( Iterator player=begin; player!=end; ++player ) {
         const size_t value( player->count() );
         if( value > maxvalue && value < 22u ) {
            maxvalue = value;
         }
      }

      for( Iterator player=begin; player!=end; ++player ) {
         if( player->count() == maxvalue )
            player->addWin();
      }

      // Reshuffling the deck of cards
      const bool shuffle( deck_.size() < 100u );
      if( shuffle ) {
         deck_.shuffle();
      }
      for( Iterator player=begin; player!=end; ++player ) {
         player->newRound( shuffle );
      }

      // Estimating a new start player
      players_.push_back( players_.front() );
      players_.erase( players_.begin() );
   }
}
//*************************************************************************************************








//=================================================================================================
//
//  NESTED CLASS CONTROLLEDPLAYER
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the ControlledPlayer class.
//
// \param player The player to be controlled.
// \return void
*/
Game::ControlledPlayer::ControlledPlayer( PlayerID player )
   : out_( false )
   , player_( player )
   , value_( 0u )
   , hand_()
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Offering the player a card.
//
// \return true in case the player want to draw another card, false if the player stands.
*/
bool Game::ControlledPlayer::offer()
{
   out_ = !player_->offer();
   return !out_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Giving the player a new card.
//
// \param card The new card to be given to the player.
// \return void
*/
void Game::ControlledPlayer::give( CardID card )
{
   player_->give( card );
   hand_.add( card );
   value_ = hand_.count();
   if( value_ > 21u ) out_ = true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Introducing another player participating in the game.
//
// \param player The other player to be introduced.
// \return void
*/
void Game::ControlledPlayer::introduce( PlayerID player )
{
   player_->introduce( player );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notifying the player that another player has drawn a card.
//
// \param player The other player who has drawn a card.
// \param card The card which has been drawn.
// \return void
*/
void Game::ControlledPlayer::playerDrawsCard( const ControlledPlayer& player, CardID card )
{
   player_->playerDrawsCard( player.player_, card );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notifying the player that another player stands.
//
// \param player The other player who stands.
// \return void
*/
void Game::ControlledPlayer::playerStands( const ControlledPlayer& player )
{
   player_->playerStands( player.player_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notifying the player of the start of a new round.
//
// \param shuffle true in case the deck is re-shuffled, false in case it is not re-shuffled.
// \return void
*/
void Game::ControlledPlayer::newRound( bool shuffle )
{
   player_->newRound( shuffle );
   out_ = false;
   value_ = 0u;
   hand_.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notifying the player of the start of a new game.
//
// \return void
*/
void Game::ControlledPlayer::newGame()
{
   player_->newGame();
}
//*************************************************************************************************

} // blackjack
