#include "bits-and-bobs.hh"

void printRanks(const std::vector<std::tuple<int, int>> &scoresWithRank) {
  for (auto &t : scoresWithRank) {
    std::cout << std::get<0>(t) << " - " << std::get<1>(t) << "\n";
  }
}

int main() {
  auto parts = readIntParts(std::cin);
  auto &rankedScores = parts.at(0);
  auto &playerScores = parts.at(1);

  std::vector<std::tuple<int, int>> scoresWithRank;
  int curRank = 1;
  int prevScore = -1;
  for (auto &score : rankedScores) {
    assert(score > 0);
    if (score != prevScore) {
      scoresWithRank.push_back(std::make_tuple(score, curRank));
      curRank++;
    }
    prevScore = score;
  }
  // printRanks(scoresWithRank);

  for (auto &playerScore : playerScores) {
  }
}
