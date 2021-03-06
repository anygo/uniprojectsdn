//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_SEVEN_H_
#define _BLACKJACK_CORE_SEVEN_H_


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
/*!\brief Implementation of a seven card.
//
// This class represents the card seven with a card value of 7. It is implemented as a veneer
// of the Card class to correctly initialize a seven card.
*/
class Seven : public Card
{
 public:
   //**Constructor*********************************************************************************
   Seven( CardColor color );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   ~Seven();
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
