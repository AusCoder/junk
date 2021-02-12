package com.sebmuellermath.algos.unionfind;

import java.util.Random;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class Percolation {
  // given a size n, create the union find problem
  // maintain array of open nodes
  // Add nodes randomly, calculate p as the number of open nodes

  private int size;
  private int[] openSites;
  private WeightedQuickUnion unionFind;

  public Percolation(int n) {
    size = n;
    openSites = new int[n * n];
    for (int i = 0; i < n * n; i++) {
      openSites[i] = 0;
    }
    unionFind = new WeightedQuickUnion(n * n + 2);
    // grid looks like:
    // 0 1 2
    // 3 4 5
    // 6 7 8
    for (int i = 0; i < n; i++) {
      unionFind.union(i, n * n);
      unionFind.union(n * (n - 1) + i, n * n + 1);
    }
  }

  public void open(int row, int col) {
    int pos = getAndValidatePosition(row, col);
    openSites[pos] = 1;
    if (col > 0 && isOpen(row, col-1)) {
      unionFind.union(pos, getAndValidatePosition(row, col-1));
    }
    if (col < size - 1 && isOpen(row, col+1)) {
      unionFind.union(pos, getAndValidatePosition(row, col+1));
    }
    if (row > 0 && isOpen(row-1, col)) {
      unionFind.union(pos, getAndValidatePosition(row-1, col));
    }
    if (row < size - 1 && isOpen(row+1, col)) {
      unionFind.union(pos, getAndValidatePosition(row+1, col));
    }
  }

  public boolean isOpen(int row, int col) {
    return openSites[getAndValidatePosition(row, col)] == 1;
  }

  public boolean isFull(int row, int col) {
    int pos = getAndValidatePosition(row, col);
    return isOpen(row, col) && unionFind.isConnected(pos, size*size);
  }

  public int numberOfOpenSites() {
    int sum = 0;
    for (int i = 0; i < openSites.length; i++) {
      sum += openSites[i];
    }
    return sum;
  }

  public boolean doesPercolate() {
    return unionFind.isConnected(size*size, size*size+1);
  }

  private int getAndValidatePosition(int row, int col) {
    if (row < 0 || row >= size || col < 0 || col >= size) {
      throw new IllegalArgumentException("Bad row or col");
    }
    return row * size + col;
  }

  public static void main(String[] args) {
    int size = 100;
    int experimentCount = 100;

    double sumResults =
      IntStream.range(0, experimentCount)
        .mapToDouble(i -> runExperiment(size))
        .reduce(0, Double::sum);
    double mean = sumResults / experimentCount;
    System.out.println(mean);
  }

  private static double runExperiment(int size) {
    Percolation percolation = new Percolation(size);
    Random rng = new Random();
    while (!percolation.doesPercolate()) {
      int pos = rng.nextInt();
      pos = (pos < 0 ? -pos : pos) % (size * size);
      int row = pos / size;
      int col = pos % size;
      if (!percolation.isOpen(row, col)) {
        percolation.open(row, col);
      }
    }
    return (double)percolation.numberOfOpenSites() / (size * size);
  }
}
