#ifndef _BITS_AND_BOBS_HH
#define _BITS_AND_BOBS_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
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

template <typename T> void printMap(const T &map) {
  for (const auto &item : map) {
    std::cout << item.first << " -> " << item.second << "  ";
  }
  std::cout << "\n";
}

template <typename T> void print(const T &thing) { std::cout << thing << "\n"; }

template <typename T>
std::vector<T> getColumn(int colIdx,
                         const std::vector<std::vector<T>> &matrix) {
  std::vector<T> col;
  std::transform(matrix.cbegin(), matrix.cend(), std::back_inserter(col),
                 [=](auto row) { return row.at(colIdx); });
  return col;
}

/*
  Read lines from a file, possibly without reading
  final empty line.
*/
std::vector<std::string> readLinesFromFile(const std::string &filePath,
                                           bool dropFinalEmptyLine = true) {
  std::ifstream iStream{filePath};
  if (!iStream.is_open()) {
    throw std::runtime_error(std::string("Failed to open ") + filePath);
  }

  std::vector<std::string> lines;
  std::string line;
  for (;;) {
    if (iStream.eof()) {
      break;
    }
    std::getline(iStream, line);
    lines.push_back(line);
  }
  if (dropFinalEmptyLine && (lines.at(lines.size() - 1).empty())) {
    lines.erase(lines.cend() - 1);
  }
  return lines;
}

std::vector<std::string> readLinesUntilEmptyLine(std::istream &iStream) {
  std::vector<std::string> lines;
  std::string line;
  for (;;) {
    std::getline(iStream, line);
    if (line.size() == 0) {
      break;
    }
    lines.push_back(line);
  }
  return lines;
}

/*
  Parse matrix from string with space separators
*/
template <typename T, typename F>
std::vector<T> parseVector(const std::string &line, F parseFn) {
  std::istringstream iss(line);
  auto startIter = std::istream_iterator<std::string>(iss);
  auto endIter = std::istream_iterator<std::string>();
  std::vector<std::string> strElems(startIter, endIter);
  std::vector<T> elems;
  std::transform(strElems.cbegin(), strElems.cend(), std::back_inserter(elems),
                 parseFn);
  return elems;
}

/*
  Parse matix from vector of strings with space separators
*/
template <typename T, typename F>
std::vector<std::vector<T>> parseMatrix(const std::vector<std::string> &input,
                                        F parseFn) {
  std::vector<std::vector<T>> matrix;
  std::transform(input.cbegin(), input.cend(), std::back_inserter(matrix),
                 [=](const auto &l) { return parseVector<T, F>(l, parseFn); });
  return matrix;
}

std::vector<std::vector<std::string>>
parseMatrixStr(const std::vector<std::string> &lines) {
  auto parseFn = [](std::string x) { return x; };
  return parseMatrix<std::string, decltype(parseFn)>(lines, parseFn);
}

std::vector<std::vector<int>>
parseMatrixInt(const std::vector<std::string> &lines) {
  auto parseFn = [](std::string x) { return std::stoi(x); };
  return parseMatrix<int, decltype(parseFn)>(lines, parseFn);
}

std::vector<std::vector<double>>
parseMatrixDouble(const std::vector<std::string> &lines) {
  auto parseFn = [](std::string x) { return std::stod(x); };
  return parseMatrix<double, decltype(parseFn)>(lines, parseFn);
}

bool isNumber(const std::string &s) {
  return std::all_of(s.cbegin(), s.cend(),
                     [](auto c) { return std::isdigit(c); });
}

#endif
