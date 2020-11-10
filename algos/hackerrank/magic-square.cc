#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

void formMagicSquare(std::vector<std::vector<int>> matrix) {}

int main() {
  std::string line;

  std::vector<std::vector<int>> matrix;

  for (;;) {
    std::getline(std::cin, line);
    if (line.size() == 0) {
      break;
    }
    std::istringstream iss(line);

    std::vector<std::string> elems((std::istream_iterator<std::string>(iss)),
                                   std::istream_iterator<std::string>());
    // XXX: Without the parens, it is a most vexing parse
    // std::vector<std::string> elems(std::istream_iterator<std::string>(iss),
    //                                std::istream_iterator<std::string>());
    std::vector<int> row;
    std::transform(elems.begin(), elems.end(), std::back_inserter(row),
                   [](auto s) { return std::stoi(s); });

    matrix.push_back(row);
  }

  // for (auto &s : row) {
  //   std::cout << s << " - ";
  // }
  // std::cout << "\n";
}
