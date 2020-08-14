#include <iostream>

template <typename T> class Thing {
private:
  T value;

public:
  explicit Thing(T v) : value{v} {}
  T getValue() const { return value; }
};

template <typename T> inline auto MakeThing(T t) { return Thing<T>(t); }

void ThingTaker(const Thing<int> &t) {
  std::cout << "Thing has value " << t.getValue() << "\n";
}

int main() {
  // Can't do this, need to specify the type in the template
  // (it works for C++17)
  // ThingTaker(Thing(3));

  // Use the helper creation function
  ThingTaker(MakeThing(3));
}