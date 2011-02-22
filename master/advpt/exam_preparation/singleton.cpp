#include <iostream>

class SingletonGoogleDocs {

		private:
				SingletonGoogleDocs() {}
				SingletonGoogleDocs(const SingletonGoogleDocs&) {}
				~SingletonGoogleDocs() {}

		public:
				static SingletonGoogleDocs* getInstance() {
	
						static SingletonGoogleDocs s;
						return &s;
				}
				void doSomething() {
		
						std::cout << "hallo" << std::endl;
				}
};

class SingletonToni {

		private:
				SingletonToni() {}
				SingletonToni(const SingletonToni&) {}
				~SingletonToni() {}

		public:
				static SingletonToni& getInstance() {
						
						static SingletonToni s;
						return s;
				}
				void doSomething() {
					
						std::cout << "der toni - der assi toni" << std::endl;
				}
};

int main(int argc, char* argv[]) {
	
		SingletonToni& toni = SingletonToni::getInstance();
		toni.doSomething();

		SingletonGoogleDocs* gdocs = SingletonGoogleDocs::getInstance();
		gdocs->doSomething();

}
