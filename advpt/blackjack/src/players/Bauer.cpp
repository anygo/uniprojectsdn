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
/*!\brief Implementation of the blackjack avatar for Martin Bauer.
//
// This class represents the blackjack avatar for Martin Bauer. It implements one of the most
// basic blackjack strategies possible: whenever the total sum of the cards Martin holds is larger
// than 18, he stands, otherwise he draws another card. This strategy is enforced, even if another
// player is already closer to 21.
*/
class Bauer : public RegisteredPlayer<Bauer>
{
 private:
   //**Type definitions****************************************************************************
   typedef RegisteredPlayer<Bauer>  Base;   // Type of the base class.
   typedef std::vector<CardID>      Cards;  // Container for all cards player Bauer holds.
   //**********************************************************************************************

 public:
   //**Constructor and destructor******************************************************************
   Bauer();
   ~Bauer();
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   virtual bool offer();
   virtual void give( CardID card );
   virtual void newRound( bool shuffle );
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Cards       cards_;       // The cards in the hand of player Bauer.
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

bool Bauer::registered_ = Bauer::registerThis();




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for the Bauer blackjack avatar.
*/
Bauer::Bauer()
   : Base( "Martin Bauer" )  // Initialization of the base class
   , cards_()                // The cards in the hand of player Bauer
   , value_()                // The total value of all cards in the hand
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the Bauer blackjack avatar..
*/
Bauer::~Bauer()
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
// \return true in case the cards have a higher value than 18, false otherwise.
*/
bool Bauer::offer()
{
   if( value_ > 18U ) return false;
   else               return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Implementation of the give function.
//
// \param card The card I'm given by the dealer.
// \return void
*/
void Bauer::give( CardID card )
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
void Bauer::newRound( bool /*shuffle*/ )
{
   value_ = 0;
   cards_.clear();
}
//*************************************************************************************************

} // blackjack
