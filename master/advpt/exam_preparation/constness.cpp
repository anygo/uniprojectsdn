#include <iostream>

class C {
		public:
				int& f() { return i_; }
				int f() const { return i_; }
				C(const int i) : i_(i) {}

		private:
				int i_;
};

int main(int argc, char* argv[]) {
	C c(1);
	const C c_const(10);

	std::cout << c.f() << std::endl;
	std::cout << c_const.f() << std::endl;
	
	c.f()++;

	// won't work:
	//c_const.f()++; 

	std::cout << c.f() << std::endl;
	std::cout << c_const.f() << std::endl;
}
