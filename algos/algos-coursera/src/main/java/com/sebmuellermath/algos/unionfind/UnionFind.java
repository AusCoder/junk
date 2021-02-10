package com.sebmuellermath.algos.unionfind;

interface UnionFind {
  public boolean isConnected(int p, int q);

  public void union(int p, int q);
}
