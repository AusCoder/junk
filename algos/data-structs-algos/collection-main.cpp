#include <iostream>
#include "collection.h"

using namespace std;

int main(int argc, char **argv)
{
    cout << "Example using unique ptr" << endl;
    CollectionUniquePrt<int> c {10};

    c.insert(1);
    c.insert(2);
    c.insert(4);
    c.remove(3);
    c.insert(3);
    c.remove(2);
    c.remove(3);

    cout << "isEmpty: " << c.isEmpty() << endl;
    cout << "contains 1: " << c.contains(1) << endl;
    cout << "contains 2: " << c.contains(2) << endl;

    CollectionUniquePrt<int> c2 {c};
    c2.insert(7);
    c.printAll();
    c2.printAll();

    cout << "****************" << endl;
    cout << "Example using new/delete" << endl;

    CollectionRaw<int> rc {10};
    rc.insert(1);
    rc.insert(2);
    rc.insert(4);
    rc.remove(3);
    rc.insert(3);
    rc.remove(2);
    rc.remove(3);

    cout << "isEmpty: " << rc.isEmpty() << endl;
    cout << "contains 1: " << rc.contains(1) << endl;
    cout << "contains 2: " << rc.contains(2) << endl;

    CollectionRaw<int> rc2 {rc};
    rc.insert(6);
    rc2.insert(7);
    rc.printAll();
    rc2.printAll();

    return 0;
}
