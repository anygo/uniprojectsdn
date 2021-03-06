//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_NINE_H_
#define _BLACKJACK_CORE_NINE_H_


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
/*!\brief Implementation of a nine card.
//
// This class represents the card nine with a card value of 9. It is implemented as a veneer
// of the Card class to correctly initialize a nine card.
*/
class Nine : public Card
{
 public:
   //**Constructor*********************************************************************************
   Nine( CardColor color );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   ~Nine();
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
