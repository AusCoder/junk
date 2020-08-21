#include <iterator>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
  std::map<std::string, int> counts{{"abc", 3}, {"def", 4}};

  // std::map requires an inserter
  std::map<std::string, int> doubledCounts;
  std::transform(counts.begin(), counts.end(),
                 std::inserter(doubledCounts, doubledCounts.begin()),
                 [](auto &x) -> std::pair<std::string, int> {return {x.first, 2 * x.second};});

  std::cout << "Doubled counts are:\n";
  for (auto &[name, count] : doubledCounts) {
    std::cout << name << ": " << count << "\n";
  }

  // std::vector seg faults if we don't have a std::inserter in the following code
  // It doesn't segault if the vector has sufficient capacity though
  std::vector<int> flatCounts{17, 89, 73};
  std::vector<int> doubledFlatCounts;
  // std::transform(flatCounts.begin(), flatCounts.end(),
  //                doubledFlatCounts.begin(),
  //                [](auto &x) {return 2 * x;});
  std::transform(flatCounts.begin(), flatCounts.end(),
                 std::inserter(doubledFlatCounts, doubledFlatCounts.begin()),
                 [](auto &x) {return 2 * x;});
  std::cout << "Doubled flat counts are:\n";
  for (auto &x : doubledFlatCounts) {
    std::cout << x << "\n";
  }
}
