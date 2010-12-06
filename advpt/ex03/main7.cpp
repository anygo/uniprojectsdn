#include <iostream>

class Complex {

		private:
				double _real, _imag;

		public:
				Complex(double real, double imag = 0.0) : 
						_real(real), _imag(imag) {}

				const Complex operator+(const Complex& other) const {

						Complex ret(_real + other._real, _imag + other._imag);
						return ret;
				}

				Complex& operator++() {

						++_real;
						return *this;
				}

				const Complex operator++(int) {

						Complex tmp = *this;	
						++_real;
						return tmp;
				}

				friend std::ostream& operator<<(std::ostream& lhs, const Complex& rhs) {
		
						lhs << "(" << rhs._real << ", " << rhs._imag << ")";
						return lhs;
				}
};


int main(int argc, char* argv[]) {
	
		// test
		Complex a(2,1);
		Complex b = a;
		Complex c = a + b;
		Complex d = c++;

		std::cout << a << " " << b << " " << c << " " << d;
}

