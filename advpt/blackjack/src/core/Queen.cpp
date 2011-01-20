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

#include <blackjack/core/Queen.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the Queen class.
//
// \param cardcolor The color of the card (heart, diamond, spade, or club).
*/
Queen::Queen( CardColor cardcolor )
   : Card( queen, cardcolor, 10u )  // Initialization of the base class
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the Queen class.
*/
Queen::~Queen()
{}
//*************************************************************************************************

} // blackjack
