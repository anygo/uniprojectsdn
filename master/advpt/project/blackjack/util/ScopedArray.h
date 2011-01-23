//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Klaus Iglberger
*/
//=================================================================================================

#ifndef _BLACKJACK_UTIL_SCOPEDARRAY_H_
#define _BLACKJACK_UTIL_SCOPEDARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstddef>
#include <blackjack/util/NonCopyable.h>


namespace blackjack {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Array managing RAII class.
//
// The ScopedArray class is a classical RAII class for the management of a dynamically allocated
// array. An array passed by either the constructor or the reset() function to a scoped array is
// destroyed the moment the scoped array goes out of scope.
*/
template< typename T >
class ScopedArray : private NonCopyable
{
 private:
   //**Type definitions****************************************************************************
   typedef ScopedArray<T>  Self;
   //**********************************************************************************************

 public:
   //**Constructor and destructor******************************************************************
   explicit inline ScopedArray( T* p = 0 );
            inline ~ScopedArray();
   //**********************************************************************************************

   //**Operators***********************************************************************************
   inline T&       operator[]( std::size_t index );
   inline const T& operator[]( std::size_t index ) const;
   //**********************************************************************************************

   //**Utiltiy functions***************************************************************************
   inline void reset( T* p = 0 ) /* throw() */;
   inline T*   get  () const /* throw() */;
   inline void swap ( ScopedArray& sa ) /* throw() */;
   //**********************************************************************************************

 private:
   //**Forbidden operations************************************************************************
   void operator==( const ScopedArray& ) const;
   void operator!=( const ScopedArray& ) const;
   //**********************************************************************************************

   //**Constructor and destructor******************************************************************
   T* p_;  // Pointer to the managed array.
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the ScopedArray class.
//
// \param p Pointer to the dynamically allocated array to be managed.
*/
template< typename T >
inline ScopedArray<T>::ScopedArray( T* p )
   : p_( p )  // Pointer to the managed array
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the ScopedArray class.
*/
template< typename T >
inline ScopedArray<T>::~ScopedArray()
{
   delete [] p_;
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the access to the array elements.
//
// \param index Access index.
// \return Reference to the accessed element.
*/
template< typename T >
inline T& ScopedArray<T>::operator[]( std::size_t index )
{
   assert( p_ != 0 );
   return p_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the access to the array elements.
//
// \param index Access index.
// \return Reference-to-const to the accessed element.
*/
template< typename T >
inline const T& ScopedArray<T>::operator[]( std::size_t index ) const
{
   assert( p_ != 0 );
   return p_[index];
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Resetting the scoped array.
//
// \param p Pointer to another dynamically allocated array.
// \return void
//
// This function resets the scoped array by destroying the currently managed array and taking
// responsibility for the given array.
*/
template< typename T >
inline void ScopedArray<T>::reset( T* p ) /* throw() */
{
   Self( p ).swap( *this );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the managed array.
//
// \return Pointer to the managed array.
*/
template< typename T >
inline T* ScopedArray<T>::get() const /* throw() */
{
   return p_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two scoped arrays.
//
// \param b The other scoped array to be swapped.
// \return void
*/
template< typename T >
inline void ScopedArray<T>::swap( ScopedArray& b ) /* throw() */
{
   T* tmp = b.p_;
   b.p_ = p_;
   p_ = tmp;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Swapping the contents of two scoped arrays.
//
// \param a The left-hand side scoped array to be swapped.
// \param b The right-hand side scoped array to be swapped.
// \return void
*/
template< typename T >
inline void swap( ScopedArray<T>& a, ScopedArray<T>& b ) /* throw() */
{
   a.swap( b );
}
//*************************************************************************************************

} // namespace blackjack

#endif
