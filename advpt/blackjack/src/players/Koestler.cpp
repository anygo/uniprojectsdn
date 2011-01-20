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
/*!\brief Implementation of the blackjack avatar for Harald Köstler.
//
// This class represents the blackjack avatar for Harald Köstler. It implements one of the most
// basic blackjack strategies possible: whenever the total sum of the cards Harald holds is larger
// than 17, he stands, otherwise he draws another card. This strategy is enforced, even if another
// player is already closer to 21.
*/
class Koestler : public RegisteredPlayer<Koestler>
{
 private:
   //**Type definitions****************************************************************************
   typedef RegisteredPlayer<Koestler>  Base;   // Type of the base class.
   typedef std::vector<CardID>         Cards;  // Container for all cards player Koestler holds.
   //**********************************************************************************************

 public:
   //**Constructor and destructor******************************************************************
   Koestler();
   ~Koestler();
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   virtual bool offer();
   virtual void give( CardID card );
   virtual void newRound( bool shuffle );
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Cards       cards_;       // The cards in the hand of player Koestler.
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

bool Koestler::registered_ = Koestler::registerThis();




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for the Koestler blackjack avatar.
*/
Koestler::Koestler()
   : Base( "Harald Köstler" )  // Initialization of the base class
   , cards_()                  // The cards in the hand of player Koestler
   , value_()                  // The total value of all cards in the hand
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the Koestler blackjack avatar..
*/
Koestler::~Koestler()
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
// \return true in case the cards have a higher value than 17, false otherwise.
*/
bool Koestler::offer()
{
   if( value_ > 17U ) return false;
   else               return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Implementation of the give function.
//
// \param card The card I'm given by the dealer.
// \return void
*/
void Koestler::give( CardID card )
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
void Koestler::newRound( bool /*shuffle*/ )
{
   value_ = 0;
   cards_.clear();
}
//*************************************************************************************************

} // blackjack
