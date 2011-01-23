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
#include <blackjack/core/CardType.h>


namespace blackjack {

//*************************************************************************************************
/*!\brief Global output operator for card types.
//
// \param os Reference to the output stream.
// \param type The card type to be added to the stream.
// \return Reference to the output stream.
*/
std::ostream& operator<<( std::ostream& os, CardType type )
{
   switch( type )
   {
      case two  : os << "Two";   break;
      case three: os << "Three"; break;
      case four : os << "Four";  break;
      case five : os << "Five";  break;
      case six  : os << "Six";   break;
      case seven: os << "Seven"; break;
      case eight: os << "Eight"; break;
      case nine : os << "Nine";  break;
      case ten  : os << "Ten";   break;
      case jack : os << "Jack";  break;
      case queen: os << "Queen"; break;
      case king : os << "King";  break;
      case ace  : os << "Ace";   break;
      default: assert( false );  break;
   }

   return os;
}
//*************************************************************************************************

} // namespace blackjack
