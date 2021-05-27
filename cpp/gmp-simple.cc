/*
  Compile with something like:

  g++ -g -Wall -Werror -lgmpxx -lgmp -o gmp-simple gmp-simple.cc
*/
#include <gmpxx.h>
#include <iostream>
#include <unordered_map>

using namespace std;

template <typename T> void printMap(const T &map) {
  for (const auto &x : map) {
    cout << x.first << " -> " << x.second << "\n";
  }
}

int main() {
  mpz_class x = 0;
  x += 100000;
  cout << x << "\n";

  // can't use mpz_class as key, it's hash seems to be undefined or
  // deleted?
  unordered_map<int, mpz_class> cache;
  cache.insert({0, 45});
  cache.insert({1, 450000000000000});
  cache.insert({0, 60});
  printMap(cache);
}
