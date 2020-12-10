#include "../../common/include/bits-and-bobs.hh"

bool isSolved(const std::vector<std::vector<int>> &matrix) {
  const int expectedSum = 15;
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

  bool sumsAreCorrect = true;
  for (int y = 0; y < 3; y++) {
    int sum = std::accumulate(matrix.at(y).begin(), matrix.at(y).end(), 0,
                              std::plus<int>());
    sumsAreCorrect &= (sum == expectedSum);
  }
  for (int x = 0; x < 3; x++) {
    int sum = 0;
    for (int y = 0; y < 3; y++) {
      sum += matrix.at(y).at(x);
    }
    sumsAreCorrect &= (sum == expectedSum);
  }
  {
    int sum = 0;
    for (int x = 0; x < 3; x++) {
      sum += matrix.at(x).at(x);
    }
    sumsAreCorrect &= (sum == expectedSum);
  }
  {
    int sum = 0;
    for (int x = 0; x < 3; x++) {
      sum += matrix.at(x).at(2 - x);
    }
    sumsAreCorrect &= (sum == expectedSum);
  }
  return hitAllNums && sumsAreCorrect;
}

bool canPlaceNumber(const std::vector<std::vector<int>> &matrix, int startPos,
                    int k) {
  for (int idx = 0; idx < startPos - 1; idx++) {
    int x = idx % 3;
    int y = idx / 3;
    if (matrix.at(y).at(x) == k) {
      return false;
    }
  }
  return true;
}

/*
  This is a back tracking algorithm to form a magic square.

  The idea is to run through every position and digit possibility
  and see what cost we get by solving the magic square from that position.
  If we don't solve the magic square, we say the cost is infinite.
  We keep track of the minimum cost so far.

  Is this a branch and bound strategy?
*/
int _formMagicSquare(std::vector<std::vector<int>> &matrix, int startPos,
                     int currentCost, int minCost) {
  for (int idx = startPos; idx < 9; idx++) {
    int x = idx % 3;
    int y = idx / 3;

    for (int k = 1; k < 10; k++) {
      if (canPlaceNumber(matrix, startPos, k)) {
        int prevValue = matrix.at(y).at(x);
        int newCost = currentCost + std::abs(prevValue - k);

        if (newCost < minCost) {
          matrix.at(y).at(x) = k;
          int minCostFromHere =
              _formMagicSquare(matrix, startPos + 1, newCost, minCost);
          if (minCostFromHere < minCost) {
            minCost = minCostFromHere;
          }
          matrix.at(y).at(x) = prevValue;
        }
      }
    }
    // At this point, I have tried all possibilities from startPos
    // onwards and know the minCost.
    return minCost;
  }
  // At this point, I have gone through the entire matrix
  if (isSolved(matrix)) {
    // std::cout << "_formMagicSquare: " << startPos << ", " << currentCost <<
    // ", "
    //           << minCost << "\n";
    // printMatrix(matrix);
    return currentCost;
  } else {
    return std::numeric_limits<int>::max();
  }
}

int formMagicSquare(std::vector<std::vector<int>> matrix) {
  return _formMagicSquare(matrix, 0, 0, std::numeric_limits<int>::max());
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
  std::cout << formMagicSquare(matrix) << "\n";
}
