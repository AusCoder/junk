#include <iostream>

/*
  Don't overload and override at the same time.
 */

class base {
public:
  virtual int foo(int x) {return 0;}
};

class derived : base {
public:
  auto foo(long x) {return 1;}
  auto foo(double x) {return 2;}

  derived() {
    // this int variant does not compile because the compiler
    // stops looking once it finds a foo, and it grabs all foos
    // at that scope
    std::cout << foo(0) << "\n";
    // these both work both these foos live in the same scope
    std::cout << foo(0L) << "\n";
    std::cout << foo(0.0) << "\n";
  }
};

int main(int argc, char *argv[]) {
  derived d{};
  return 0;
}
