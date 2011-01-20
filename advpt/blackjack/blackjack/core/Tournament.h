//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_TOURNAMENT_H_
#define _BLACKJACK_CORE_TOURNAMENT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blackjack/util/NonCopyable.h>


namespace blackjack {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of the blackjack tournament organization.
//
// This class represents the black jack tournament event. It implements the scheduling of games
// and players according to the tournament settings (see the file blackjack/config/Tournament.h).
*/
class Tournament : private NonCopyable
{
 public:
   //**Constructor/destructor**********************************************************************
   // No excplicitly declared constructor.
   // No excplicitly declared destructor.
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   void run();
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
