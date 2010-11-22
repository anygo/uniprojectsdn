//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 9
//
//  Modify the program of 'main7.cpp' such that the two-dimensional array is instead linearized
//  and allocated as an one-dimensional array. Print the array to resemble the output of
//  'main7.cpp'. Then copy the contents of the array to a second linearized array and print it.
//  Also repeat the valgrind test.
//
//=================================================================================================


#include <iostream>
using namespace std;


int main(int argc, char* argv[]) {

		int m,n;

		cout << "Dimensionen angeben: " << std::endl;
		cin >> m >> n;

		int *matrix = new int[m*n];

		for(int i = 0; i < m; ++i)
			for(int j = 0; j < n; ++j)
				matrix[i*m+j] = i*m + j;

		cout << "Original: " << endl;
		for(int i = 0; i < m; ++i) {	
			for(int j = 0; j < n; ++j) { 
				cout << matrix[i*m+j] << " ";
			}
			cout << endl;
		}


		int *matrix2 = new int[m*n];
		copy(matrix,matrix+m*n,matrix2);
		
		cout << "Kopie: " << std::endl;
		for(int i = 0; i < m; ++i) {	
			for(int j = 0; j < n; ++j) { 
				cout << matrix2[i*m+j] << " ";
			}
			cout << endl;
		}

		delete [] matrix;
		delete [] matrix2;

		return 0;
}
