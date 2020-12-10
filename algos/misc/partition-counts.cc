/*
  A partition of a positive integer p is a set of
  strictly positive integers whose sum is p.

  Let p(n) denote the number of partitions of integer
  n, then this exercise is about computing p(n) using
  a recursive formula.

  From daily programmer 2020-10-21 challenge #386
  https://www.reddit.com/r/dailyprogrammer/comments/jfcuz5/20201021_challenge_386_intermediate_partition/
*/
#include "../../common/include/bits-and-bobs.hh"
#include <gmpxx.h>

#define MAX_CACHE_SIZE 1000000

typedef mpz_class seq_t;

/*
  The sign sequence in the partition formula.
*/
int signSequence(std::unordered_map<int, int> &cache, int n) {
  if (cache.size() > MAX_CACHE_SIZE) {
    print("WARNING: sign cache size is large");
  }
  int seqValue = 1;
  if (cache.find(n) != cache.end()) {
    seqValue = cache.at(n);
  } else {
    seqValue = 1;
    for (int i = 0; i < n; i++) {
      // diff is the alternating counting and odds sequence
      // 1 3 2 5 3 7 4 9 5 11
      int diff = i % 2 == 0 ? i / 2 + 1 : i + 2;
      seqValue += diff;
    }
    cache.insert({n, seqValue});
  }
  return seqValue;
}

/*
  Calculate the partition sequence at position n.
*/
seq_t partitionSequence(std::unordered_map<int, seq_t> &partitionCache,
                        std::unordered_map<int, int> &signCache, int n) {
  if (partitionCache.size() > MAX_CACHE_SIZE) {
    print("WARNING: partition cache size is large");
  }
  seq_t result = 0;
  if (partitionCache.find(n) != partitionCache.end()) {
    result = partitionCache.at(n);
  } else if (n == 0) {
    result = 1;
  } else {
    result = 0;
    for (int i = 0; i < n; i++) {
      int signIdx = signSequence(signCache, i);
      if (n - signIdx < 0) {
        break;
      } else {
        int sign = (-1 * ((i / 2) % 2)) * 2 + 1;
        seq_t p =
            sign * partitionSequence(partitionCache, signCache, n - signIdx);
        result += p;
      }
    }
    partitionCache.insert({n, result});
  }
  return result;
}

int main() {
  std::unordered_map<int, seq_t> partitionCache;
  std::unordered_map<int, int> signCache;
  auto seqValue = partitionSequence(partitionCache, signCache, 666);
  std::cout << "Sequence value: " << seqValue << "\n";
  // check if it seems right
  int digitLength = 0;
  int digitSum = 0;
  while (seqValue > 0) {
    digitLength++;
    mpz_class mod;
    (seqValue % 10).eval(mod.get_mpz_t());
    digitSum += mod.get_si();
    seqValue /= 10;
  }
  std::cout << "Digit length: " << digitLength << "\n"
            << "Digit sum: " << digitSum << "\n";
}
