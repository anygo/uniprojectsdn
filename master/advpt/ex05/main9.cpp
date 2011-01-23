//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 5 - Task 9
//
//  Consider the implementation of a three-dimensional vector class in 'main9.cpp'. With
//  this implementation it is possible to add two vectors of the same data type. However, the
//  attempt to add two vectors of different data type (for instance an integer vector and a
//  'double' vector) results in a compilation error. Extend the functionality such that it is
//  possible to add two vectors of different data types. The return type of the addition
//  operation should be a vector of the higher order data type (for example 'Vector3<double>'
//  in case a 'Vector3<int>' and a 'Vector3<double' are added).
//
//=================================================================================================


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cassert>
#include <cstddef>
#include <iostream>
using std::size_t;


//*************************************************************************************************
// Definition of the Vector3 class
template< typename Type >
class Vector3
{
 public:
   // Default constructor
   Vector3()
   {
      v_[0] = v_[1] = v_[2] = Type();
   }

   // Constructor for a homogenous initialization of all elements
   Vector3( const Type& init )
   {
      v_[0] = v_[1] = v_[2] = init;
   }

   // Constructor for a direct initialization of all vector elements
   Vector3( const Type& x, const Type& y, const Type& z )
   {
      v_[0] = x;
      v_[1] = y;
      v_[2] = z;
   }

   // Copy constructor
   Vector3( const Vector3& vector )
   {
      v_[0] = vector[0];
      v_[1] = vector[1];
      v_[2] = vector[2];
   }

   // Copy assignment operator
   Vector3& operator=( const Vector3& vector )
   {
      v_[0] = vector[0];
      v_[1] = vector[1];
      v_[2] = vector[2];
   }

   // Data access functions
   Type&       operator[]( size_t index )       { assert( index < 3U ); return v_[index]; }
   const Type& operator[]( size_t index ) const { assert( index < 3U ); return v_[index]; }

 private:
   Type v_[3];  // The three statically allocated vector elements.
};
//*************************************************************************************************


//*************************************************************************************************
// Vector3 addition operator
template< typename Type >
const Vector3<Type> operator+( const Vector3<Type>& lhs, const Vector3<Type>& rhs )
{
   return Vector3<Type>( lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2] );
}
//*************************************************************************************************


//*************************************************************************************************
// Vector3 output operator
template< typename Type >
std::ostream& operator<<( std::ostream& os, const Vector3<Type>& vector )
{
   return os << "<" << vector[0] << "," << vector[1] << "," << vector[2] << ">";
}
//*************************************************************************************************


//*************************************************************************************************
int main( int argc, char** argv )
{
   // Addition of two integer vectors
   {
      const Vector3<int> a( 1, 2, 3 );
      const Vector3<int> b( 1, 2, 3 );

      const Vector3<int> c( a + b );

      std::cout << " c = " << c << "\n";
   }

   // Addition of an integer and a double vector
   {
      const Vector3<int>    a( 1  , 2  , 3   );
      const Vector3<double> b( 1.1, 2.2, 3.3 );

      const Vector3<double> c( a + b );

      std::cout << " c = " << c << "\n";
   }
}
//*************************************************************************************************
