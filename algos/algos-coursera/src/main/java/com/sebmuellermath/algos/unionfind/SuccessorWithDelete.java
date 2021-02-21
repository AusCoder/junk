package com.sebmuellermath.algos.unionfind;
/*
Given a set of integers S of form {0 ... n-1}, write routines
that support:
  - removing an integer x
  - finding the smallest element of S greater or equal to x
*/

public class SuccessorWithDelete {
  private int size;
  private MaxElementQuickUnion unionFind;

  public SuccessorWithDelete(int n) {
    size = n;
    unionFind = new MaxElementQuickUnion(size + 1);
  }

  public void remove(int x) {
    // forbit removing the last element
    if (x < 0 || x >= size - 1) {
      throw new IllegalArgumentException("out of range");
    }
    unionFind.union(x, x + 1);
  }

  public int successor(int x) {
    if (x < 0 || x >= size) {
      throw new IllegalArgumentException("out of range");
    }
    return unionFind.find(x);
  }

  public static void main(String[] args) {
    SuccessorWithDelete succ = new SuccessorWithDelete(5);
    succ.remove(1);
    succ.remove(3);
    succ.remove(0);
    succ.remove(2);
    System.out.println(succ.successor(0));
    System.out.println(succ.successor(1));
    System.out.println(succ.successor(2));
    System.out.println(succ.successor(3));
    System.out.println(succ.successor(4));
  }
}
