//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Neumann
*/
//=================================================================================================


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <string>
#include <vector>
#include <blackjack/core/Algorithm.h>
#include <blackjack/core/Card.h>
#include <blackjack/core/RegisteredPlayer.h>
#include <blackjack/core/Types.h>


namespace blackjack {

//*************************************************************************************************
/*!\brief Implementation of the blackjack avatar for Dominik Neumann.
// This is my implementation ;)
*/
class Neumann : public RegisteredPlayer<Neumann>
{
 private:
   //**Type definitions****************************************************************************
   typedef RegisteredPlayer<Neumann>    Base;   // Type of the base class.
   typedef std::vector<CardID>          Cards;  // Container for all cards player Neumann holds.
   //**********************************************************************************************

 public:
   //**Constructor and destructor******************************************************************
   Neumann();
   ~Neumann();
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   virtual bool offer();
   virtual void give( CardID card );
   virtual void newRound( bool shuffle );
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Cards       cards_;       // The cards in the hand of player Neumann.
   value_t     value_;       // The total value of all cards in the hand.
   static bool registered_;  // Registration flag.
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
//
//=================================================================================================

bool Neumann::registered_ = Neumann::registerThis();




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for the Neumann blackjack avatar.
*/
Neumann::Neumann()
   : Base( "Dominik Neumann" )  // Initialization of the base class
   , cards_()                   // The cards in the hand of player Neumann
   , value_()                   // The total value of all cards in the hand
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the Neumann blackjack avatar..
*/
Neumann::~Neumann()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of the offer function.
//
// \return true in case my cards have a higher value than 16, false otherwise.
*/
bool Neumann::offer()
{
   if( value_ > 16U ) return false;
   else               return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Implementation of the give function.
//
// \param card The card I'm given by the dealer.
// \return void
*/
void Neumann::give( CardID card )
{
   cards_.push_back( card );
   value_ = count( cards_.begin(), cards_.end() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Implementation of the newRound function.
//
// \param shuffle true in case the deck of cards is re-shuffled, false otherwise.
// \return void
*/
void Neumann::newRound( bool /*shuffle*/ )
{
   value_ = 0;
   cards_.clear();
}
//*************************************************************************************************

} // blackjack
