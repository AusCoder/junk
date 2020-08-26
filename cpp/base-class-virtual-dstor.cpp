#include <iostream>
#include <memory>

class BaseNonVirtualDstor {
public:
  virtual int logic() = 0;
};

class DerivedNonVirtualDstor : public BaseNonVirtualDstor {
public:
  ~DerivedNonVirtualDstor() {
    std::cout << "Destroying important stuff!\n";
  }

  int logic() override {
    std::cout << "Important logic happening!\n";
    return 1;
  }
};

class BaseVirtualDstor {
public:
  virtual int logic() = 0;
  virtual ~BaseVirtualDstor() = default;
};

class DerivedVirtualDstor : public BaseVirtualDstor {
public:
  ~DerivedVirtualDstor() override {
    std::cout << "Destroying important stuff!\n";
  }

  int logic() override {
    std::cout << "Important logic happening!\n";
    return 1;
  }
};


int main() {
  std::cout << "Without the virtual dstor\n";
  auto dnvdPtr = std::make_unique<DerivedNonVirtualDstor>();
  dnvdPtr->logic();
  BaseNonVirtualDstor *bnvdPtr{new DerivedNonVirtualDstor()};
  bnvdPtr->logic();
  delete bnvdPtr;

  std::cout << "With the virtual dstor\n";
  auto dvdPtr = std::make_unique<DerivedVirtualDstor>();
  dvdPtr->logic();
  BaseVirtualDstor *bvdPtr{new DerivedVirtualDstor()};
  bvdPtr->logic();
  delete bvdPtr;
}
