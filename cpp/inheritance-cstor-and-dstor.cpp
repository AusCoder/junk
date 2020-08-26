#include <iostream>

using namespace std;

class BaseNonVirt {
public:
    BaseNonVirt() {
        cout << "BaseNonVirt Constructor Called\n";
    }
    ~BaseNonVirt() {
        cout << "BaseNonVirt Destructor called\n";
    }
};

class DerivedNonVirt: public BaseNonVirt {
public:
    DerivedNonVirt(){
        cout << "DerivedNonVirt constructor called\n";
    }
    ~DerivedNonVirt(){
        cout << "DerivedNonVirt destructor called\n";
    }
};

class BaseVirt {
public:
    BaseVirt() {
        cout << "BaseVirt Constructor Called\n";
    }
    virtual ~BaseVirt() {
        cout << "BaseVirt Destructor called\n";
    }
};

class DerivedVirt: public BaseVirt {
public:
    DerivedVirt(){
        cout << "DerivedVirt constructor called\n";
    }
    ~DerivedVirt(){
        cout << "DerivedVirt destructor called\n";
    }
};


int main() {
  cout << ">> Without a virtual base dstor\n";
  BaseNonVirt *bnv = new DerivedNonVirt();
  delete bnv;

  cout << ">> With a virtual base dstor\n";
  BaseVirt *bv = new DerivedVirt();
  delete bv;
}
