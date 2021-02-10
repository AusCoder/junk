package com.sebmuellermath.algos.unionfind;

import java.util.stream.IntStream;

public class QuickUnion implements UnionFind {
  private int[] ids;

  public QuickUnion(int n) {
    ids = new int[n];
    for (int i = 0; i < ids.length; i++) {
      ids[i] = i;
    }
  }

  @Override
  public boolean isConnected(int p, int q) {
    return getRoot(p) == getRoot(q);
  }

  @Override
  public void union(int p, int q) {
    int pRoot = getRoot(p);
    int qRoot = getRoot(q);
    ids[pRoot] = qRoot;
  }

  private int getRoot(int p) {
    while (ids[p] != p) {
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
}
