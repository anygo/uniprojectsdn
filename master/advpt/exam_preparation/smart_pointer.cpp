#include <iostream>
#include <memory>

class C {
	public:
			C() : arr_(new int[10]) {}
			~C() { delete [] arr_; }

	private:
			int* arr_;
};

class D {
	public:
			D() : arr_(new int[10]) {}
			
	private:
			std::auto_ptr<int> arr_;
};

void f() {
	C c;
	D d;
}

int main(int argc, char* argv[]) {
	f();
	int a = 1;
	a++;
}
