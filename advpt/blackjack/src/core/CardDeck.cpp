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

#include <cassert>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <blackjack/core/Ace.h>
#include <blackjack/core/CardColor.h>
#include <blackjack/core/CardDeck.h>
#include <blackjack/core/Eight.h>
#include <blackjack/core/Five.h>
#include <blackjack/core/Four.h>
#include <blackjack/core/Jack.h>
#include <blackjack/core/King.h>
#include <blackjack/core/Nine.h>
#include <blackjack/core/Queen.h>
#include <blackjack/core/Seven.h>
#include <blackjack/core/Six.h>
#include <blackjack/core/Ten.h>
#include <blackjack/core/Three.h>
#include <blackjack/core/Two.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the CardDeck class.
//
// The constructor initializes the deck of cards and immediately shuffles the deck three times.
*/
CardDeck::CardDeck()
   : cards_()  // Container for all cards contained in the deck
   , top_()    // Topmost card of the card deck.
{
   cards_.reserve( 312 );

   // Adding six card games into the deck
   for( int i=0; i<6; ++i )
   {
      // Adding thirteen heart cards
      cards_.push_back( new Two  ( heart ) );
      cards_.push_back( new Three( heart ) );
      cards_.push_back( new Four ( heart ) );
      cards_.push_back( new Five ( heart ) );
      cards_.push_back( new Six  ( heart ) );
      cards_.push_back( new Seven( heart ) );
      cards_.push_back( new Eight( heart ) );
      cards_.push_back( new Nine ( heart ) );
      cards_.push_back( new Ten  ( heart ) );
      cards_.push_back( new Jack ( heart ) );
      cards_.push_back( new Queen( heart ) );
      cards_.push_back( new King ( heart ) );
      cards_.push_back( new Ace  ( heart ) );

      // Adding thirteen diamond cards
      cards_.push_back( new Two  ( diamond ) );
      cards_.push_back( new Three( diamond ) );
      cards_.push_back( new Four ( diamond ) );
      cards_.push_back( new Five ( diamond ) );
      cards_.push_back( new Six  ( diamond ) );
      cards_.push_back( new Seven( diamond ) );
      cards_.push_back( new Eight( diamond ) );
      cards_.push_back( new Nine ( diamond ) );
      cards_.push_back( new Ten  ( diamond ) );
      cards_.push_back( new Jack ( diamond ) );
      cards_.push_back( new Queen( diamond ) );
      cards_.push_back( new King ( diamond ) );
      cards_.push_back( new Ace  ( diamond ) );

      // Adding thirteen spade cards
      cards_.push_back( new Two  ( spade ) );
      cards_.push_back( new Three( spade ) );
      cards_.push_back( new Four ( spade ) );
      cards_.push_back( new Five ( spade ) );
      cards_.push_back( new Six  ( spade ) );
      cards_.push_back( new Seven( spade ) );
      cards_.push_back( new Eight( spade ) );
      cards_.push_back( new Nine ( spade ) );
      cards_.push_back( new Ten  ( spade ) );
      cards_.push_back( new Jack ( spade ) );
      cards_.push_back( new Queen( spade ) );
      cards_.push_back( new King ( spade ) );
      cards_.push_back( new Ace  ( spade ) );

      // Adding thirteen club cards
      cards_.push_back( new Two  ( club ) );
      cards_.push_back( new Three( club ) );
      cards_.push_back( new Four ( club ) );
      cards_.push_back( new Five ( club ) );
      cards_.push_back( new Six  ( club ) );
      cards_.push_back( new Seven( club ) );
      cards_.push_back( new Eight( club ) );
      cards_.push_back( new Nine ( club ) );
      cards_.push_back( new Ten  ( club ) );
      cards_.push_back( new Jack ( club ) );
      cards_.push_back( new Queen( club ) );
      cards_.push_back( new King ( club ) );
      cards_.push_back( new Ace  ( club ) );
   }

   // Shuffling three times
   shuffle();
   shuffle();
   shuffle();

   // Setting the iterator to the topmost card
   top_ = cards_.begin();
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the CardDeck class.
*/
CardDeck::~CardDeck()
{
   for( Iterator card=cards_.begin(); card!=cards_.end(); ++card )
      delete *card;
   cards_.clear();
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Drawing a card from the deck of cards.
//
// \return The topmost card of the deck of cards.
// \exception std::logic_error Drawing a card from an empty deck.
*/
CardID CardDeck::draw()
{
   if( top_ == cards_.end() )
      throw std::logic_error( "Drawing a card from an empty deck" );
   return *(top_++);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reshuffling the entire card deck.
//
// \return void
//
// This function reshuffles the entire card deck including the cards that have already been
// drawn from the deck. Afterwards, the deck has a size of 312.
*/
void CardDeck::shuffle()
{
   // Shuffling the cards
   for( int i=0; i<10000; ++i ) {
      std::swap( cards_[rand()%312], cards_[rand()%312] );
   }

   // Resetting the iterator to the topmost card
   top_ = cards_.begin();

   assert( size() == 312u );
}
//*************************************************************************************************

} // blackjack
