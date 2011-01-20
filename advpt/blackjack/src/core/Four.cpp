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

#include <blackjack/core/Four.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the Four class.
//
// \param cardcolor The color of the card (heart, diamond, spade, or club).
*/
Four::Four( CardColor cardcolor )
   : Card( four, cardcolor, 4u )  // Initialization of the base class
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the Four class.
*/
Four::~Four()
{}
//*************************************************************************************************

} // blackjack
