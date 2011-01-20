//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_CARDTYPE_H_
#define _BLACKJACK_CORE_CARDTYPE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iosfwd>


namespace blackjack {

//=================================================================================================
//
//  CARD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Card type enumeration.
//
// This enumeration represents all possible card types from a two to an ace.
*/
enum CardType {
   two   = 0,   // Code for rank two cards.
   three = 1,   // Code for three two cards.
   four  = 2,   // Code for four two cards.
   five  = 3,   // Code for five two cards.
   six   = 4,   // Code for six two cards.
   seven = 5,   // Code for seven two cards.
   eight = 6,   // Code for eight two cards.
   nine  = 7,   // Code for nine two cards.
   ten   = 8,   // Code for ten two cards.
   jack  = 9,   // Code for jacks.
   queen = 10,  // Code for queens.
   king  = 11,  // Code for kings.
   ace   = 12   // Code for aces.
};
//*************************************************************************************************




//=================================================================================================
//
//  CARD TYPE UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
std::ostream& operator<<( std::ostream& os, CardType type );
//*************************************************************************************************

} // namespace blackjack

#endif
