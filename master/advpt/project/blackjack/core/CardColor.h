//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_CARDCOLOR_H_
#define _BLACKJACK_CORE_CARDCOLOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iosfwd>


namespace blackjack {

//=================================================================================================
//
//  CARD COLORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Card color enumeration.
//
// This enumeration represents the four card colors heart, diamond, spade, and club.
*/
enum CardColor {
   heart   = 0,  // Code for the color 'heart'.
   diamond = 1,  // Code for the color 'diamond'.
   spade   = 2,  // Code for the color 'spade'.
   club    = 3   // Code for the color 'club'.
};
//*************************************************************************************************




//=================================================================================================
//
//  CARD COLOR UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
std::ostream& operator<<( std::ostream& os, CardColor color );
//*************************************************************************************************

} // namespace blackjack

#endif
