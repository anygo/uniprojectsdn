//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 7
//
//  Write a program to create a dynamically allocated, two-dimensional array filled with integers
//  of your choice, where the dimensions of the array should be given by the user on the command
//  line. Print the two-dimensional array in a matrix-like fashion. Then copy the values from the
//  array into a second two-dimensional array of the same dimensions and print it. Test for
//  invalid memory accesses and proper deletion of the memory with the tool valgrind (for a
//  valgrind tutorial, see the AdvPT web page).
//
//=================================================================================================


#include <iostream>
using namespace std;


int main(int argc, char* argv[]) {

		int m,n;

		cout << "Dimensionen angeben: " << std::endl;
		cin >> m >> n;

		int **matrix = new int*[m];
		for(int i = 0; i < m; ++i)
				matrix[i] = new int[n];

		for(int i = 0; i < m; ++i)
			for(int j = 0; j < n; ++j)
				matrix[i][j] = i*m + j;

		cout << "Original: " << endl;
		for(int i = 0; i < m; ++i) {	
			for(int j = 0; j < n; ++j) { 
				cout << matrix[i][j] << " ";
			}
			cout << endl;
		}


		int **matrix2 = new int*[m];
		for(int i = 0; i < m; ++i) {
			matrix2[i] = new int[n];
			copy(matrix[i],matrix[i]+n,matrix2[i]);
		}
		
		cout << "Kopie: " << std::endl;
		for(int i = 0; i < m; ++i) {	
			for(int j = 0; j < n; ++j) { 
				cout << matrix[i][j] << " ";
			}
			cout << endl;
		}

		for(int i = 0; i < m; ++i) {	
			delete [] matrix[i];
			delete [] matrix2[i];	
		}

		delete [] matrix;
		delete [] matrix2;

		return 0;
}
