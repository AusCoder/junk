#include <list>
#include <iostream>

using namespace std;

template<class A>
void exercise1_printByIndexList(const list<A> l, const list<int> p)
{
    auto lIter = l.begin();
    int prev = 0;
    for (auto &x: p) {
        int inc = x - prev;
        prev = x;
        for (int i = 0; i < inc; i++) {
            lIter++;
        }
        cout << *lIter << " ";
    }
    cout << endl;
}

template<class A>
void incWhileLessThan(
    typename list<A>::const_iterator &iter,
    const typename list<A>::const_iterator end,
    A cmp)
{
    while (iter != end && *iter < cmp) {
        iter++;
    }
}

template<class A>
list<A> exercise4_intersectionSorted(const list<A> &l1, const list<A> &l2)
{
    list<A> result {};
    auto l1Iter = l1.begin();
    auto l2Iter = l2.begin();

    while (l1Iter != l1.end() && l2Iter != l2.end())
    {
        auto h1 = *l1Iter;
        auto h2 = *l2Iter;
        if (h1 < h2) {
            incWhileLessThan(l1Iter, l1.end(), h2);
        } else if (h1 > h2) {
            incWhileLessThan(l2Iter, l2.end(), h1);
        } else {
            result.push_back(h1);
            l1Iter++;
            l2Iter++;
        }
    }

    return result;
}

template<class A>
void pushWhileLessThan(
    typename list<A>::const_iterator &iter,
    const typename list<A>::const_iterator end,
    A cmp,
    list<A> &output)
{
    while (iter != end && *iter < cmp) {
        output.push_back(*iter++);
    }
}

template<class A>
list<A> exercise5_unionSorted(const list<A> &l1, const list<A> &l2)
{
    list<A> result {};
    auto l1Iter = l1.begin();
    auto l2Iter = l2.begin();

    while (l1Iter != l1.end() && l2Iter != l2.end())
    {
        auto h1 = *l1Iter;
        auto h2 = *l2Iter;
        if (h1 < h2) {
            pushWhileLessThan(l1Iter, l1.end(), h2, result);
        } else if (h1 > h2) {
            pushWhileLessThan(l2Iter, l2.end(), h1, result);
        } else {
            result.push_back(h1);
            l1Iter++;
            l2Iter++;
        }
    }
    while (l1Iter != l1.end()) {
        result.push_back(*l1Iter++);
    }
    while (l2Iter != l2.end()) {
        result.push_back(*l2Iter++);
    }

    return result;
}

int main(int argc, char *argv[])
{
    list<int> l1 {1, 2, 3, 4, 5};
    list<int> l2 {-1, 4, 5, 7, 9};

    cout << "Exercise 1:" << endl;
    exercise1_printByIndexList(l1, {1, 2, 4});

    cout << "Exercise 4:" << endl;
    auto exercise4Result = exercise4_intersectionSorted(l1, l2);
    for (auto &x : exercise4Result) {
        cout << x << endl;
    }

    cout << "Exercise 5:" << endl;
    auto exercise5Result = exercise5_unionSorted(l1, l2);
    for (auto &x : exercise5Result) {
        cout << x << endl;
    }
    return 1;
}
