package com.sebmuellermath.algos.queue;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Queue<A> implements Iterable<A> {
  private int tail;
  private int size;
  private int initialCapacity = 2;
  private A[] arr;

  public Queue() {
    tail = 0;
    size = 0;
    arr = (A []) new Object[initialCapacity];
  }

  public void enqueue(A a) {
    if (size >= arr.length) {
      resize(arr.length * 2);
    }
    arr[(tail + size) % arr.length] = a;
    size++;
  }

  public A dequeue() {
    if (size == 0) {
      throw new NoSuchElementException();
    }
    A x = arr[tail];
    arr[tail++] = null;
    tail %= arr.length;
    size--;
    return x;
  }

  private void resize(int newSize) {
    A[] newArr = (A []) new Object[newSize];
    for (int i = 0; i < size; i++) {
      newArr[i] = arr[(tail + i) % arr.length];
    }
    arr = newArr;
    tail = 0;
  }

  @Override
  public Iterator<A> iterator() {
    return new QueueIterator();
  }

  private class QueueIterator implements Iterator<A> {
    private int iterIdx = tail;

    @Override
    public boolean hasNext() {
      return iterIdx < tail + size;
    }

    @Override
    public A next() {
      return arr[iterIdx++ % arr.length];
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
