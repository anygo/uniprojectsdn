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

#include <ostream>
#include <blackjack/core/Card.h>


namespace blackjack {

//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the Card class.
//
// \param cardtype The type of the card.
// \param cardcolor The color of the card.
// \param cardvalue The value of the card.
*/
Card::Card( CardType cardtype, CardColor cardcolor, value_t cardvalue )
   : type_ ( cardtype  )  // Type of the card (two, three, ..., jack, queen, ...)
   , color_( cardcolor )  // Color of the card (heart, spade, diamond, club)
   , value_( cardvalue )  // Value of the card (2 to 11)
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the Card class.
*/
Card::~Card()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Output of the card data.
//
// \param os Reference to the output stream.
// \return void
*/
void Card::print( std::ostream& os ) const
{
   os << "(" << type_ << "," << color_ << ",value=" << value_ << ")";
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Global output operator for cards.
//
// \param os Reference to the output stream.
// \param card Reference to a constant card object.
// \return Reference to the output stream.
*/
std::ostream& operator<<( std::ostream& os, const Card& card )
{
   card.print( os );
   return os;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global output operator for card handles.
//
// \param os Reference to the output stream.
// \param card Constant card handle.
// \return Reference to the output stream.
*/
std::ostream& operator<<( std::ostream& os, ConstCardID card )
{
   card->print( os );
   return os;
}
//*************************************************************************************************

} // blackjack
