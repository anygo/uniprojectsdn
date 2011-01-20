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

#include <blackjack/core/Player.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the Player class.
//
// \param fullname The full name of the player.
*/
Player::Player( const std::string& fullname )
   : name_( fullname )  // Name of the player
   , wins_( 0u )        // Accumulated number of wins
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the Player class.
*/
Player::~Player()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Introducing another player participating in the same game.
//
// \param player The other player to be introcuded.
//
// Before every new game, all players participating in the same game are introduced to each
// other. This is done by using this function, which can be used to remember all other players
// participating in the game.
*/
void Player::introduce( PlayerID /*player*/ )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notification of another player drawing a card.
//
// \param player The other player drawing a card.
// \param card The card drawn by the other player.
//
// This function notifies the player that another player has drawn a specific card. Via this
// function it is for example possible to remember which player holds which cards in his/her
// hands.
*/
void Player::playerDrawsCard( PlayerID /*player*/, CardID /*card*/ )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notification of another player deciding to stand.
//
// \param player The other player deciding to stand.
//
// This function notifies the player that another player has decided to stand, i.e., not to take
// another card. This function can be used to optimize the card drawing strategy.
*/
void Player::playerStands( PlayerID /*player*/ )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notification of the beginning of a new round.
//
// \param shuffle true in case the deck of cards is re-shuffled, false otherwise.
//
// This function signals the beginning of a new round, i.e., new cards are drawn. In case the
// shuffle parameter is true, the deck has been shuffled prior to this new round, otherwise no
// re-shuffling has taken place.
*/
void Player::newRound( bool /*shuffle*/ )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Notification of the beginning of a new game.
//
// This function signals the beginning of a new game, i.e., new involved players and a new
// deck of cards.
*/
void Player::newGame()
{}
//*************************************************************************************************

} // blackjack
