#include <iostream>

/* This works in C++17
 * In C++14, we try to use the deleted move cstor
 */

struct S {
  S() = default;
  S(const S &) = delete;
  S(S &&) = delete;

  int value = 0;
};

auto s_factory() {
  return S{};
}

int main() {
  auto s = s_factory();
  std::cout << "s value is " << s.value << "\n";
}
