//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_CARD_H_
#define _BLACKJACK_CORE_CARD_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iosfwd>
#include <blackjack/core/CardColor.h>
#include <blackjack/core/CardType.h>
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
/*!\brief Implementation of a black jack card.
//
// The Card class represents a single card of a deck of cards.
*/
class Card : private NonCopyable
{
 public:
   //**Constructor*********************************************************************************
   Card( CardType type, CardColor color, value_t cardvalue );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   ~Card();
   //**********************************************************************************************

 public:
   //**Get functions*******************************************************************************
   inline CardType  type () const { return type_;  }
   inline CardColor color() const { return color_; }
   inline value_t   value() const { return value_; }
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   void print( std::ostream& os ) const;
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   const CardType  type_;   // Type of the card (two, three, ..., jack, queen, ...)
   const CardColor color_;  // Color of the card (heart, spade, diamond, club)
   const value_t   value_;  // Value of the card (2 to 11)
   //**********************************************************************************************

   //**Member variables****************************************************************************
   friend class CardDeck;
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
std::ostream& operator<<( std::ostream& os, const Card& card );
std::ostream& operator<<( std::ostream& os, ConstCardID card );
//*************************************************************************************************

} // namespace blackjack

#endif
