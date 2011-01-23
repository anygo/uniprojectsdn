//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_ALGORITHM_H_
#define _BLACKJACK_CORE_ALGORITHM_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blackjack/core/CardType.h>
#include <blackjack/util/Types.h>


namespace blackjack {

//=================================================================================================
//
//  GENERIC COUNT FUNCTION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Calculating the total value of a range of cards.
//
// \param begin Iterator to the first card of the card range.
// \param end Iterator one past the end of the card range.
// \return The total value of the range of cards.
*/
template< typename Iterator >
value_t count( Iterator begin, Iterator end )
{
   count_t aces ( 0u );
   value_t value( 0u );

   for( ; begin!=end; ++begin ) {
      value += (*begin)->value();
      if( (*begin)->type() == ace )
         ++aces;
   }

   while( value > 21u && aces > 0u ) {
      value -= 10u;
      aces  -= 1u;
   }

   return value;
}
//*************************************************************************************************

} // namespace blackjack

#endif
