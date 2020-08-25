#include <cassert>
#include <iostream>
#include <memory>

/**
 * Take away: make any methods you want to override virtual,
 * otherwise the virtual function table dispatch thing doesn't
 * get triggered.
 *
 * If you add override on the overriding methods on the derived class,
 * you will get a compiler error if the base method is not virtual!
 * Take away: use override!
 */

class BaseNonVirtual {
public:
  int erroneous() { return 0; }
};

class DerivedNonVirtual : public BaseNonVirtual {
public:
  int erroneous() { return 1; }
  void testDerived() { // can't use override here!
    assert(erroneous() == 1);
    std::cout << "success in derived class \n";
  }
};

class BaseVirtual {
public:
  virtual int erroneous() { return 0; }
};

class DerivedVirtual : public BaseVirtual {
public:
  int erroneous() override { return 1; }
  void testDerived() {
    assert(erroneous() == 1);
    std::cout << "success in derived class \n";
  }
};

int main() {
  auto derivedNonVirtualPtr = std::make_unique<DerivedNonVirtual>();
  derivedNonVirtualPtr->testDerived();
  BaseNonVirtual *baseNonVirtualPtr{derivedNonVirtualPtr.get()};
  if (baseNonVirtualPtr->erroneous() != 1) {
    std::cout << "failed with base non virtual pointer\n";
  } else {
    assert(false);
  }

  auto derivedVirtualPtr = std::make_unique<DerivedVirtual>();
  derivedVirtualPtr->testDerived();
  BaseVirtual *baseVirtualPtr{derivedVirtualPtr.get()};
  if (baseVirtualPtr->erroneous() == 1) {
    std::cout << "success with base virtual pointer\n";
  } else {
    assert(false);
  }
}