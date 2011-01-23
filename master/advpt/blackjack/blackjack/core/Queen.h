//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_QUEEN_H_
#define _BLACKJACK_CORE_QUEEN_H_


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
/*!\brief Implementation of a queen card.
//
// This class represents the a queen with a card value of 10. It is implemented as a veneer
// of the Card class to correctly initialize a queen card.
*/
class Queen : public Card
{
 public:
   //**Constructor*********************************************************************************
   Queen( CardColor color );
   //**********************************************************************************************

 protected:
   //**Destructor**********************************************************************************
   ~Queen();
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif