//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 4 - Task 10
//
//  Define a 'SparseVector' class based on the 'std::map' associative container, which
//  represents a sparse vector for double precision data values. Additionally, define an
//  according operator+, which allows the addition of two sparse vectors.
//  Think about all aspects of how to properly design the sparse vector class (copy control,
//  representation of the vector elements, access to the vector elements, ...). The
//  'SparseVector' class should have a 'size' function to acquire the current size of the
//  vector, a 'resize' function to change the size of the vector, and it should provide an
//  overloaded subscript operator to access the vector elements (think of a suitable range
//  check!). Additionally, for an efficient traversal of the non-zero elements of the sparse
//  vector, provide 'begin' and 'end' functions that return according iterators over the
//  non-zero elements.
//  Demonstrate your implementation by means of a simple 'main' function that adds to sparse
//  vectors and prints the calculation result to the screen.
//
//=================================================================================================



