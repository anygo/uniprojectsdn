//=================================================================================================
//
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2010
//  Assignment 1 - Task 8
//
//  Rewrite the previous program (main7.cpp), but use 'vector's instead of plain arrays.
//
//=================================================================================================


#include <iostream>
#include <vector>
using namespace std;


int main(int argc, char* argv[]) {

		int m,n;

		cout << "Dimensionen angeben: " << std::endl;
		cin >> m >> n;

		vector<vector<int> > mat(m, vector<int>(n,42));

		cout << "Original: " << endl;
		for(int i = 0; i < m; ++i) {	
			for(int j = 0; j < n; ++j) { 
				cout << mat[i][j] << " ";
			}
			cout << endl;
		}

		vector<vector<int> > mat2(mat);
		
		cout << "Kopie: " << endl;
		for(int i = 0; i < m; ++i) {	
			for(int j = 0; j < n; ++j) { 
				cout << mat2[i][j] << " ";
			}
			cout << endl;
		}


		return 0;
}
