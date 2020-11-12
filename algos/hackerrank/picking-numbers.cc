#include "bits-and-bobs.hh"

int pickingNumbers(std::vector<int> arr) {
  std::unordered_map<int, int> digitCounts;
  for (auto &digit : arr) {
    if (digitCounts.find(digit) == digitCounts.cend()) {
      digitCounts.insert({digit, 0});
    }
    digitCounts.at(digit)++;
  }

  std::vector<int> digits;
  std::transform(digitCounts.cbegin(), digitCounts.cend(),
                 std::back_inserter(digits), [](auto &it) { return it.first; });
  std::sort(digits.begin(), digits.end());

  int bestSubarraySize = 0;
  for (auto it = digits.cbegin(); it != digits.cend(); it++) {
    int subarraySize = digitCounts.at(*it);
    if (subarraySize > bestSubarraySize) {
      bestSubarraySize = subarraySize;
    }
    if ((it + 1 != digits.cend()) && (*(it + 1) == *it + 1)) {
      int subarraySize = digitCounts.at(*it) + digitCounts.at(*(it + 1));
      if (subarraySize > bestSubarraySize) {
        bestSubarraySize = subarraySize;
      }
    }
  }
  return bestSubarraySize;
}

int main() {
  auto parts = readIntParts(std::cin);
  assert(parts.size() == 2);
  int bestSubarraySize = pickingNumbers(parts.at(1));
  std::cout << bestSubarraySize << "\n";
}
