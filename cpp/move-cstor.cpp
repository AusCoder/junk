#include <iostream>
#include <sstream>
#include <string>
#include <utility>

class Thing {
public:
  Thing(int v) : value{v} {}
  Thing(const Thing &) = delete;
  Thing &operator=(const Thing &) = delete;
  // Thing(Thing &&other) {}
  // Thing &operator=(Thing &&other) {}
  ~Thing() {}

private:
  int value;
};

class ThingPrivateCstor {
public:
  static ThingPrivateCstor MakeThingPrivateCstor(int v) {
    // This requires a move or copy cstor
    return ThingPrivateCstor(v);
  }

  ThingPrivateCstor(const ThingPrivateCstor &) = delete;
  ThingPrivateCstor &operator=(const ThingPrivateCstor &) = delete;
  ThingPrivateCstor(ThingPrivateCstor &&other) {
    std::swap(value, other.value);
  };
  ThingPrivateCstor &operator=(ThingPrivateCstor &&other) {
    std::swap(value, other.value);
    return *this;
  };
  ~ThingPrivateCstor() {}

  std::string render() {
    std::stringstream ss;
    ss << "ThingPrivateCstor(value=" << value << ")";
    return ss.str();
  }

private:
  ThingPrivateCstor(int v) : value{v} {}
  int value;
};

struct HoldsAThingPrivateCstor {
  HoldsAThingPrivateCstor(int v)
      : value{ThingPrivateCstor::MakeThingPrivateCstor(v)} {}

  ThingPrivateCstor value;
};

int main() {
  HoldsAThingPrivateCstor x{2};
  std::cout << "x value is: " << x.value.render() << "\n";
  auto t = ThingPrivateCstor::MakeThingPrivateCstor(5);
  std::cout << "t is: " << t.render() << "\n";
  x.value = std::move(t);
  std::cout << "x value after move is: " << x.value.render() << "\n";
  std::cout << "t after move is: " << t.render() << "\n";
}