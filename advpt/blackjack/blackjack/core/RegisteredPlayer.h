//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_CORE_REGISTEREDPLAYER_H_
#define _BLACKJACK_CORE_REGISTEREDPLAYER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <string>
#include <blackjack/core/Player.h>
#include <blackjack/core/PlayerPool.h>


namespace blackjack {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Impementation of an automatically registered black jack player.
//
// The RegisteredPlayer class represents a player which is automatically registered for the
// black jack tournament. Every player needs to derive from this class in order to be able
// to participate in the tournament. The following example demonstrates the steps that have
// to be taken in order to properly derive from this class and to be automatically registered
// by means of the player 'Iglberger':

   \code
   class Iglberger : public RegisteredPlayer<Iglberger>
   {
    public:
      // ...

    private:
      // ...
      static bool registered_;  // Registration flag.
   };

   bool Iglberger::registered_ = Iglberger::registerThis();
   \endcode

// Step number one is to define a static member varible (in this example called registered)
// that is (as the second step) initialized by a call to the registerThis() function that
// the class inherits from the RegisteredPlayer base class. These two steps automatically
// register the player 'Iglberger' for the black jack tournament.
*/
template< typename T >
class RegisteredPlayer : public Player
{
 protected:
   //**Type definitions****************************************************************************
   typedef RegisteredPlayer<T>  Base;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   inline RegisteredPlayer( const std::string& fullname );
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   inline ~RegisteredPlayer() {}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   static inline bool registerThis();
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the RegisteredPlayer class.
//
// \param fullname The full name of the player to be registered.
*/
template< typename T >
inline RegisteredPlayer<T>::RegisteredPlayer( const std::string& fullname )
   : Player( fullname )  // Initialization of the base class
{}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Helper function for the automatic registration of the registered player.
//
// \return true
//
// This function is part of the automatic registration process of the RegisteredPlayer class.
// The function is based on the singleton pattern: it creates a single instance of the player
// to be registered and adds it to the memory pool.
*/
template< typename T >
inline bool RegisteredPlayer<T>::registerThis()
{
   static T player;
   PlayerPool::pushBack( &player );
   return true;
}
//*************************************************************************************************

} // namespace blackjack

#endif
