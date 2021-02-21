package com.sebmuellermath.algos.unionfind;
/*
A union find implementation where we have an
extra method find that returns the largest
element in a connected component.
*/

import java.util.List;
import java.util.Arrays;
import java.util.stream.IntStream;

public class MaxElementQuickUnion implements UnionFind {
  private int size;
  private int[] ids;

  MaxElementQuickUnion(int n) {
    size = n;
    ids = new int[size];
    for (int i = 0; i < size; i++) {
      ids[i] = i;
    }
  }

  @Override
  public boolean isConnected(int p, int q) {
    return getRoot(p) == getRoot(q);
  }

  @Override
  public void union(int p, int q) {
    /* we don't worry about sizes,
    in getRoot we flatten the tree a lot anyway.
    */
    int pRoot = getRoot(p);
    int qRoot = getRoot(q);
    if (pRoot != qRoot) {
      if (pRoot > qRoot) {
        ids[qRoot] = pRoot;
      } else {
        ids[pRoot] = qRoot;
      }
    }
  }

  public int find(int x) {
    return getRoot(x);
  }

  private int getRoot(int p) {
    while (p != ids[p]) {
      ids[p] = ids[ids[p]];
      p = ids[p];
    }
    return p;
  }

  public int depth() {
    return IntStream.range(0, ids.length).map(x -> depth(x)).max().orElse(0);
  }

  private int depth(int p) {
    int n = 1;
    while (ids[p] != p) {
      n++;
      p = ids[p];
    }
    return n;
  }

  public static void main(String[] args) {
    List<Integer> nums = Arrays.asList(0,1,2,3,4);
    MaxElementQuickUnion unionFind = new MaxElementQuickUnion(nums.size());
    for (int i = 0; i < nums.size() - 1; i++) {
      unionFind.union(nums.get(i), nums.get(i+1));
    }
    System.out.println(unionFind.find(0));
  }
}
