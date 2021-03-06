//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_JACK_H_
#define _BLACKJACK_CORE_JACK_H_


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
/*!\brief Implementation of a jack card.
//
// This class represents the a jack with a card value of 10. It is implemented as a veneer
// of the Card class to correctly initialize a jack card.
*/
class Jack : public Card
{
 public:
   //**Constructor*********************************************************************************
   Jack( CardColor color );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   ~Jack();
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
