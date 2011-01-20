//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_UTIL_NONCOPYABLE_H_
#define _BLACKJACK_UTIL_NONCOPYABLE_H_


namespace blackjack {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base class for non-copyable class instances.
//
// The NonCopyable class is intended to work as a base class for non-copyable classes. Both the
// copy constructor and the copy assignment operator are declared private and left undefined in
// order to prohibit copy operations of the derived classes.\n
//
// \b Note: It is not necessary to publicly derive from this class. It is sufficient to derive
// privately to prevent copy operations on the derived class.

   \code
   class A : private noncopyable
   { ... };
   \endcode
*/
class NonCopyable
{
 protected:
   //**Constructor and destructor******************************************************************
   inline NonCopyable()  {}  // Default constructor for the NonCopyable class.
   inline ~NonCopyable() {}  // Destructor of the NonCopyable class.
   //**********************************************************************************************

 private:
   //**Copy constructor and copy assignment operator***********************************************
   NonCopyable( const NonCopyable& );             // Copy constructor (private & undefined)
   NonCopyable& operator=( const NonCopyable& );  // Copy assignment operator (private & undefined)
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blackjack

#endif
