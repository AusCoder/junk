package com.sebmuellermath.algos.unionfind;

import java.util.stream.Stream;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

public class UnionFindTest {

  @ParameterizedTest
  @MethodSource
  public void initialShouldBeDisconnected(UnionFind unionFind) {
    Assertions.assertFalse(unionFind.isConnected(0,1));
  }

  private static Stream<UnionFind> initialShouldBeDisconnected() {
    return Stream.of(
      new QuickFind(2),
      new QuickUnion(2),
      new WeightedQuickUnion(2),
      new MaxElementQuickUnion(2)
    );
  }

  @ParameterizedTest
  @MethodSource
  public void shouldConnect2Nodes(UnionFind unionFind) {
    unionFind.union(0,1);
    Assertions.assertTrue(unionFind.isConnected(0,1));
  }

  private static Stream<UnionFind> shouldConnect2Nodes() {
    return Stream.of(
      new QuickFind(2),
      new QuickUnion(2),
      new WeightedQuickUnion(2),
      new MaxElementQuickUnion(2)
    );
  }

  @ParameterizedTest
  @MethodSource
  public void shouldConnect3Nodes(UnionFind unionFind) {
    unionFind.union(0,1);
    unionFind.union(1,2);
    Assertions.assertTrue(unionFind.isConnected(0,2));
  }

  private static Stream<UnionFind> shouldConnect3Nodes() {
    return Stream.of(
      new QuickFind(3),
      new QuickUnion(3),
      new WeightedQuickUnion(3),
      new MaxElementQuickUnion(3)
    );
  }

  @ParameterizedTest
  @MethodSource
  public void maxElementQuickUnionShouldFindMaxElement(List<Integer> nums) {
    MaxElementQuickUnion unionFind = new MaxElementQuickUnion(nums.size());
    for (int i = 0; i < nums.size() - 1; i++) {
      unionFind.union(nums.get(i), nums.get(i+1));
    }
    Assertions.assertTrue(unionFind.find(nums.get(0)) == 4);
  }

  private static Stream<List<Integer>> maxElementQuickUnionShouldFindMaxElement() {
    return Stream.of(
      Arrays.asList(0,1,2,3,4),
      Arrays.asList(4,3,2,1,0),
      Arrays.asList(3,2,4,1,0)
    );
  }
}
