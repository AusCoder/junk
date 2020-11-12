#ifndef _BITS_AND_BOBS_HH
#define _BITS_AND_BOBS_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

template <typename T> void printVector(const std::vector<T> &vec) {
  for (auto &x : vec) {
    std::cout << x << " ";
  }
  std::cout << "\n";
}

template <typename T>
void printMatrix(const std::vector<std::vector<T>> &matrix) {
  for (auto &vec : matrix) {
    printVector(vec);
  }
}

template <typename T, typename F>
std::vector<std::vector<T>> readParts(std::istream &iStream, F parseFn) {
  std::vector<std::vector<T>> parts;
  std::string line;

  for (;;) {
    std::getline(std::cin, line);
    if (line.size() == 0) {
      break;
    }
    std::istringstream iss(line);

    auto startIter = std::istream_iterator<std::string>(iss);
    auto endIter = std::istream_iterator<std::string>();
    std::vector<std::string> strElems(startIter, endIter);
    std::vector<T> elems;
    std::transform(strElems.begin(), strElems.end(), std::back_inserter(elems),
                   parseFn);

    parts.push_back(std::move(elems));
  }
  return parts;
}

std::vector<std::vector<std::string>> readStringParts(std::istream &iStream) {
  auto parseFn = [](std::string x) { return x; };
  return readParts<std::string, decltype(parseFn)>(iStream, parseFn);
}

std::vector<std::vector<int>> readIntParts(std::istream &iStream) {
  auto parseFn = [](std::string x) { return std::stoi(x); };
  return readParts<int, decltype(parseFn)>(iStream, parseFn);
}

#endif
