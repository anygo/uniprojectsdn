#include <iostream>

class Complex {
		
		public:
				Complex() : real_(0.f), imag_(0.f) {}
				explicit Complex(double real, double imag = 0.f) : real_(real), imag_(imag) {}

				Complex& operator++() {
						
						std::cout << "operator++() called" << std::endl;
						++real_;
						return *this;
				}

				Complex operator++(int) {
						
						std::cout << "operator++(int) called" << std::endl;
						Complex tmp = *this;
						++real_;
						return tmp;
				}

		private:
				double real_;
				double imag_;

		friend std::ostream& operator<<(std::ostream&, Complex&);
};

std::ostream& operator<<(std::ostream& os, Complex& c) {

		os << "(" << c.real_ << ", " << c.imag_ << ")";
		return os;
}


int main(int argc, char* argv[]) {

		Complex c1;
		std::cout << "c1: " << c1 << std::endl;

		Complex c2(42.f, 23.f);
		std::cout << "c2: " << c2 << std::endl;
		
		Complex c3(10.f);
		std::cout << "c3: " << c3 << std::endl;
		
		// won't work (explicit constructor!)
		//Complex c4 = 1.f;
		//std::cout << "c4: " << c4 << std::endl;

		std::cout << c1 << " -> " << ++c1 << std::endl;
		// wtf??std::cout << c1 << " -> " << c1++ << std::endl;
		++c1;
}
