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

#include <string>
#include <vector>
#include <blackjack/core/Algorithm.h>
#include <blackjack/core/Card.h>
#include <blackjack/core/RegisteredPlayer.h>
#include <blackjack/core/Types.h>


namespace blackjack {

//*************************************************************************************************
/*!\brief Implementation of the blackjack avatar for Klaus Iglberger.
//
// This class represents the blackjack avatar for me (Klaus Iglberger). It implements one of the
// most basic blackjack strategies possible: whenever the total sum of the cards I hold is larger
// than 16, I stand, otherwise I draw another card. This strategy is enforced, even if another
// player is already closer to 21.
*/
class Iglberger : public RegisteredPlayer<Iglberger>
{
 private:
   //**Type definitions****************************************************************************
   typedef RegisteredPlayer<Iglberger>  Base;   // Type of the base class.
   typedef std::vector<CardID>          Cards;  // Container for all cards player Iglberger holds.
   //**********************************************************************************************

 public:
   //**Constructor and destructor******************************************************************
   Iglberger();
   ~Iglberger();
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   virtual bool offer();
   virtual void give( CardID card );
   virtual void newRound( bool shuffle );
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Cards       cards_;       // The cards in the hand of player Iglberger.
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

bool Iglberger::registered_ = Iglberger::registerThis();




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for the Iglberger blackjack avatar.
*/
Iglberger::Iglberger()
   : Base( "Klaus Iglberger" )  // Initialization of the base class
   , cards_()                   // The cards in the hand of player Iglberger
   , value_()                   // The total value of all cards in the hand
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the Iglberger blackjack avatar..
*/
Iglberger::~Iglberger()
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
bool Iglberger::offer()
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
void Iglberger::give( CardID card )
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
void Iglberger::newRound( bool /*shuffle*/ )
{
   value_ = 0;
   cards_.clear();
}
//*************************************************************************************************

} // blackjack
