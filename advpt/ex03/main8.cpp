//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 3 - Task 8
//
//  Write a program to add two matrices and to print the result of the matrix addition to the
//  screen and into a file. Both operands are specified via an input file of the following
//  format:
//
//   8 12
//   1.5
//   2.4
//   -8.3
//   0.5
//   ...
//
//  The first line of the input file specifies the number of rows and columns (in that order)
//  of the matrix. Each subsequent line of the file specifies one double precision element of
//  the matrix in row-wise ordering.
//  Implement the program by means of a 'Matrix' class. Think about all aspects of how to
//  properly design the matrix class (copy control, representation of the matrix elements,
//  access to the matrix elements, ...). The 'Matrix' class should have a 'rows' and a
//  'columns' function to acquire the current number of rows and columns, a 'resize' function
//  to change the size of the matrix, and it should provide an overloaded function operator
//  to access the matrix elements in a 2D fashion (think of a suitable range check!). The
//  matrix addition should be realized by implementing an overloaded 'operator+'.
//  Take care to design your program such that a suitable error message is printed to the
//  screen for all possible error cases (invalid matrix sizes, errors in the input file, ...).
//  The program should be callable as e.g. './main8 lhs.dat rhs.dat result.dat', where
//  'lhs.dat' and 'rhs.dat' are the two input files for the operands of the matrix addition,
//  and 'result.dat' is the name of the file for the resulting matrix.
//
//=================================================================================================



