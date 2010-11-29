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

#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;

class Matrix {

	protected:
		int rows_;
		int cols_;
		double* data;
	

	public: 
		Matrix(int r, int c) : rows_(r), cols_(c) {
			data = new double[r*c];
		}

		// destructor
		~Matrix() {
			delete[] data;
		}
	
		// copy-constructor
		Matrix(const Matrix &m) {
			rows_ = m.rows();
			cols_ = m.cols();
		
			for(int r = 0; r < m.rows(); ++r) {
				for(int c = 0; c < m.cols(); ++c) {
					(*this)(r,c) = m(r,c);
				}
			}
		}

		inline int rows() const { return rows_; }
		inline int cols() const { return cols_; }
	
		void resize(const int r, const int c) {
			
			delete[] data;
			data = new double[r*c];
		}

		Matrix operator+(const Matrix& other) {
			
			if (!(rows() == other.rows()) || !(cols() == other.cols()) ) {
				std::cout << "Matrix dimensions must agree" << std::endl;
			}
			
			Matrix bla(rows(), cols());
			for(int r = 0; r < rows(); ++r) {
				for(int c = 0; c < cols(); ++c) {
					bla(r,c) = (*this)(r,c) + other(r,c);
				}
			}
		
			return bla;
		}

		double& operator()(const int& y, const int& x) {
			
			return data[y*cols() + x];
		}
		
		double& operator()(const int& y, const int& x) const {
			
			return data[y*cols() + x];
		}
};

int main(int argc, char* argv[]) {

	int m, n = 0;
	
	ifstream stream(argv[1]);
	stream >> m >> n;
	Matrix lhs(m,n);
	for(int r = 0; r < m; ++r) {
		for(int c = 0; c < n; ++c) {
			stream >> lhs(r,c);
		}
	}
	stream.close();
	

	ifstream stream2(argv[2]);
	stream2 >> m >> n;
	Matrix rhs(m,n);
	for(int r = 0; r < m; ++r) {
		for(int c = 0; c < n; ++c) {
			stream2 >> rhs(r,c);
		}
	}
	stream2.close();

	Matrix result = lhs + rhs;
	
	// print to file
	ofstream o(argv[3]);
	o << result.rows() << " " << result.cols() << std::endl;
	for (int r = 0; r < result.rows(); ++r) {
		for (int c = 0; c < result.cols(); ++c) {
			o << result(r,c) << std::endl;
		}
	}
	o.close();

	return 0;
}
