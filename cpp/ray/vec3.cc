#include "vec3.hh"
#include <iostream>

int main() {
  vec3 x{3, 0, 0};
  vec3 y{1, 1, 1};
  auto v = x + y;
  // double x = v[0];
  std::cout << "success " << v.length() << "\n";
}
