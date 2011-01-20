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
#include <ostream>
#include <blackjack/core/CardColor.h>


namespace blackjack {

//*************************************************************************************************
/*!\brief Global output operator for card colors.
//
// \param os Reference to the output stream.
// \param color The card color to be added to the stream.
// \return Reference to the output stream.
*/
std::ostream& operator<<( std::ostream& os, CardColor color )
{
   switch( color )
   {
      case heart  : os << "Heart";   break;
      case diamond: os << "Diamond"; break;
      case spade  : os << "Spade";   break;
      case club   : os << "Club";    break;
      default: assert( false );      break;
   }

   return os;
}
//*************************************************************************************************

} // namespace blackjack
