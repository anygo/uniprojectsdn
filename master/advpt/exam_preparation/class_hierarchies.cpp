#include <iostream>


class Base {
	public:
			virtual void f() { std::cout << "Base::f()" << std::endl; }
			virtual void g() = 0;
};

class Derived : public Base {
	public:
			virtual void f() { std::cout << "Derived::f()" << std::endl; }
			virtual void g() { std::cout << "Derived::g()" << std::endl; }
};

class Derived2 : public Derived {
	public:
			virtual void f() { std::cout << "Derived2::f()" << std::endl; }
			virtual void g() { std::cout << "Derived2::g()" << std::endl; }
			void h() { std::cout << "Derived2::h()" << std::endl; }
};

int main(int argc, char* argv[]) {

	// won't work (pure virtual!)
	//Base *b0 = new Base();

	Base *b  = new Derived();
	Base *b2 = new Derived2();
	Derived2 *d2 = new Derived2();

	b->f();
	b->g();

	b2->f();
	b2->g();
	
	// won't work:
	//b2->h();
	d2->h();
	
}
