package com.sebmuellermath.algos.stack;

import java.lang.Iterable;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class Stack<A> implements Iterable<A> {
  private int idx;
  private A[] arr;

  public Stack() {
    int initialCapacity = 2;
    arr = (A[]) new Object[initialCapacity];
  }

  public boolean isEmpty() {
    return idx == 0;
  }

  public void push(A s) {
    if (idx >= arr.length) {
      resize(2 * arr.length);
    }
    arr[idx++] = s;
  }

  public A pop() {
    if (idx <= 0) {
      throw new NoSuchElementException();
    }
    A out = arr[--idx];
    arr[idx] = null;
    // Shrink the array if we get to 25% used
    if (idx > 0 && idx == arr.length / 4) {
      resize(arr.length / 2);
    }
    return out;
  }

  public A peek() {
    if (idx <= 0) {
      throw new NoSuchElementException();
    }
    return arr[idx - 1];
  }

  @Override
  public Iterator<A> iterator() {return new StackIterator();}

  private void resize(int newCapacity) {
    A[] newArr = (A[])new Object[newCapacity];
    int copyCount = Math.min(arr.length, newCapacity);
    for (int i = 0; i < copyCount; i++) {
      newArr[i] = arr[i];
    }
    arr = newArr;
  }

  private class StackIterator implements Iterator<A> {
    private int iterIdx = idx;

    @Override
    public boolean hasNext() {return iterIdx > 0;}

    @Override
    public A next() {
      if (iterIdx <= 0) {
        throw new NoSuchElementException();
      }
      return arr[--iterIdx];
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
