//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_CARDDECK_H_
#define _BLACKJACK_CORE_CARDDECK_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iosfwd>
#include <vector>
#include <blackjack/core/Card.h>
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
/*!\brief Implementation of a deck of cards.
//
// This class represents a deck of cards for black jack containing a total of 312 (6*52) cards.
// The deck provides the following utility functions:
//  - size() to check the current size of the deck of cards
//  - draw() to draw a card from the deck; in case the deck is empty, a std::logic_error exception
//           is thrown
//  - shuffle() to reshuffle the deck; afterwards the size of the deck is 
*/
class CardDeck : private NonCopyable
{
 private:
   //**Type definitions****************************************************************************
   typedef std::vector<CardID>    Cards;          // Container type for the card deck.
   typedef Cards::iterator        Iterator;       // Iterator over non-constant cards.
   typedef Cards::const_iterator  ConstIterator;  // Iterator over constant cards.
   //**********************************************************************************************

 public:
   //**Constructor and destructor******************************************************************
   CardDeck();
   ~CardDeck();
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   inline size_t size() const;
          CardID draw();
          void   shuffle();
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Cards    cards_;  // Container for all cards contained in the deck.
   Iterator top_;    // Topmost card of the card deck.
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current size of the deck of cards.
//
// \return The current size of the deck of cards.
*/
inline size_t CardDeck::size() const
{
   return cards_.end() - top_;
}
//*************************************************************************************************

} // namespace blackjack

#endif
