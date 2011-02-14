#include <iostream>

class Parameters {

		public:
				static Parameters* get_instance() {
						static Parameters p;
						return &p;
				}
				int& get() { return i_; }
				int get() const { return i_; }
				void set(int i) { i_ = i; }

		private:
				// standard constructor -> private
				Parameters() : i_(0) {}
				// copy constructor -> private
				Parameters(const Parameters&) {}
				// copy assignment operator -> private
				Parameters& operator=(const Parameters&) {}
				
				// stored value
				int i_;
};

int main(int argc, char* argv[]) {
		
		Parameters *p1 = Parameters::get_instance();
		Parameters *p2 = Parameters::get_instance();

		p1->set(10);
		std::cout << p1->get() << std::endl;
		std::cout << p2->get() << std::endl;
		
		p1->get() = 25;
		std::cout << p1->get() << std::endl;
		std::cout << p2->get() << std::endl;
		
		p2->set(50);
		p1->get()++;
		std::cout << p1->get() << std::endl;
		std::cout << p2->get() << std::endl;
}
