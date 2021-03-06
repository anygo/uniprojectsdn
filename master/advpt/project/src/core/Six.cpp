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

#include <blackjack/core/Six.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the Six class.
//
// \param cardcolor The color of the card (heart, diamond, spade, or club).
*/
Six::Six( CardColor cardcolor )
   : Card( six, cardcolor, 6u )  // Initialization of the base class
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the Six class.
*/
Six::~Six()
{}
//*************************************************************************************************

} // blackjack
