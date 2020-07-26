#include <iostream>
#include "list.hpp"

using namespace std;

template <typename Object>
void printList(const List<Object> &l)
{
    for (const auto &x: l)
    {
        cout << x << " ";
    }
    cout << endl;
}

int main()
{
    List<int> l1{};
    l1.push_back(1);
    l1.push_back(2);
    l1.push_back(3);
    cout << "List 1: ";
    printList(l1);
    l1.pop_front();
    cout << "List 1: ";
    printList(l1);

    cout << endl;

    List<int> l2{l1};
    l2.push_front(4);
    cout << "List 1: ";
    printList(l1);
    cout << "List 2: ";
    printList(l2);

    cout << endl;

    List<int> l3{std::move(l2)};
    cout << "List 3: ";
    printList(l3);

    cout << endl;

    // This is copy cstor because first definition;
    List<int> l4 = l3;
    cout << "List 4: ";
    printList(l4);

    cout << endl;

    List<int> l5{};
    l5 = l3;
    cout << "List 5: ";
    printList(l5);

    cout << endl;

    List<int> l6{};
    l6 = std::move(l3);
    cout << "List 6: ";
    printList(l6);

    cout << endl;

    cout << "success" << endl;
}
