#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

// [
//   [9, 1, 5], // sum 15
//   [8, 4, 3], // sum 15
//   [], // sum 15
// ]

void printMatrix(const std::vector<std::vector<int>> &matrix) {
  for (auto &row : matrix) {
    for (auto &x : row) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
}

bool isSolved(const std::vector<std::vector<int>> &matrix) {
  std::array<bool, 9> hits;
  std::fill(hits.begin(), hits.end(), false);

  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 3; x++) {
      hits.at(matrix[y][x] - 1) = true;
    }
  }
  bool hitAllNums = true;
  for (auto &x : hits) {
    hitAllNums &= x;
  }
  return hitAllNums;
}

void formMagicSquare(std::vector<std::vector<int>> matrix) {
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 3; x++) {
      // if isSolved(matrix): return 0;
      // else:
      //     want to try and set a number
      // for n in range(1, 10):
      //   cost = abs(matrix[y][x] - n)
      //   matrix[y][x] = n
      //   return cost
    }
  }
}

int main() {
  std::string line;

  std::vector<std::vector<int>> matrix;

  for (;;) {
    std::getline(std::cin, line);
    if (line.size() == 0) {
      break;
    }
    std::istringstream iss(line);

    auto startIter = std::istream_iterator<std::string>(iss);
    auto endIter = std::istream_iterator<std::string>();
    std::vector<std::string> elems(startIter, endIter);
    // XXX: Without the parens, it is a most vexing parse
    // std::vector<std::string> elems(std::istream_iterator<std::string>(iss),
    //                                std::istream_iterator<std::string>());
    std::vector<int> row;
    std::transform(elems.begin(), elems.end(), std::back_inserter(row),
                   [](auto s) { return std::stoi(s); });

    matrix.push_back(row);
  }
  printMatrix(matrix);
  std::cout << isSolved(matrix) << "\n";
  // for (auto &s : matrix[0]) {
  //   std::cout << s << " - ";
  // }
  // std::cout << "\n";
}
