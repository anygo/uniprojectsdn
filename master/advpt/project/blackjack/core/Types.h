//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_TYPES_H_
#define _BLACKJACK_CORE_TYPES_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstddef>


namespace blackjack {

//=================================================================================================
//
//  ::blackjack NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

class Card;
class Player;




//=================================================================================================
//
//  TYPE DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Handle to a single, non-constant card.
*/
typedef Card*  CardID;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Handle to a single, constant card.
*/
typedef const Card*  ConstCardID;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Handle to a single, non-constant player.
*/
typedef Player*  PlayerID;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Handle to a single, constant player.
*/
typedef const Player*  ConstPlayerID;
//*************************************************************************************************

} // namespace blackjack

#endif