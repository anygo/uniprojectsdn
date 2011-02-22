#include <iostream>
using namespace std;

class Base {
public:
	 virtual void print( int i1, int i2 = 5 ) { cout << "Base" << i1 << ", " << i2 << endl; }

};

class Derived : public Base {
public:
	void print( int i1, int i2 = 77) {cout << "Derived:" << i1 << ", " << i2 << endl; }
};

int main()
{
	Base *pb = new Derived;
	Derived *d = new Derived;
	pb->print(5555); //output: Derived: 5555, 5
	d->print(1111); //output: Derived: 1111, 77
	return 0;
}
