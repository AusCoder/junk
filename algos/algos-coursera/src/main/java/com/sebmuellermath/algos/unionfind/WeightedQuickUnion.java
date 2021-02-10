package com.sebmuellermath.algos.unionfind;

import java.util.stream.IntStream;

public class WeightedQuickUnion implements UnionFind {
  private int[] ids;
  private int[] sizes;

  public WeightedQuickUnion(int n) {
    ids = new int[n];
    sizes = new int[n];
    for (int i = 0; i < ids.length; i++) {
      ids[i] = i;
      sizes[i] = 1;
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
    if (pRoot != qRoot) {
      if (sizes[pRoot] < sizes[qRoot]) {
        ids[pRoot] = qRoot;
        sizes[qRoot] += sizes[pRoot];
      } else {
        ids[qRoot] = pRoot;
        sizes[pRoot] += sizes[qRoot];
      }
    }
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
