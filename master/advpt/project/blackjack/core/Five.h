//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_FIVE_H_
#define _BLACKJACK_CORE_FIVE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blackjack/core/Card.h>


namespace blackjack {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a five card.
//
// This class represents the card five with a card value of 5. It is implemented as a veneer
// of the Card class to correctly initialize a five card.
*/
class Five : public Card
{
 public:
   //**Constructor*********************************************************************************
   Five( CardColor color );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   ~Five();
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
