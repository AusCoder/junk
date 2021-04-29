package com.sebmuellermath.algos.queue;

import java.util.Random;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class RandomizedQueue<A> implements Iterable<A> {
    private int initialCapacity = 2;
    private A[] arr;
    private int size;
    Random rng;

    public RandomizedQueue() {
        arr = (A[])new Object[initialCapacity];
        size = 0;
        rng = new Random();
    }

    public int size() {
        return size;
    }

    public void enqueue(A item) {
        if (size >= arr.length) {
            resize(size * 2);
        }
        arr[size] = item;
        size++;
    }

    private void resize(int newSize) {
        A[] newArr = (A[])new Object[newSize];
        for (int i = 0; i < arr.length; i++) {
            newArr[i] = arr[i];
        }
        arr = newArr;
    }

    public A dequeue() {
        int idx = sampleIdx();
        A tmp = arr[idx];
        arr[idx] = arr[size - 1];
        arr[size - 1] = null;
        size--;
        return tmp;
    }

    public A sample() {
        int idx = sampleIdx();
        return arr[idx];
    }

    private int sampleIdx() {
        if (size == 0) {
            throw new NoSuchElementException();
        }
        int idx = rng.nextInt(size);
        return idx;
    }

    @Override
    public Iterator<A> iterator() {
        return new RandomizedQueueIterator();
    }

    private class RandomizedQueueIterator implements Iterator<A> {
        private int[] idxs;
        private int idx = 0;

        public RandomizedQueueIterator() {
            idxs = new int[size];
            for (int i = 0; i < size; i++) {
                idxs[i] = i;
            }
            Random rng = new Random();
            for (int i = 0; i < size; i++) {
                int j = rng.nextInt(i + 1);
                int tmp = idxs[i];
                idxs[i]=idxs[j];
                idxs[j] = tmp;
            }
        }

        @Override
        public boolean hasNext() {
            return idx < idxs.length;
        }

        @Override
        public A next() {
            if (idx >= idxs.length) {
                throw new NoSuchElementException();
            }
            A tmp = arr[idxs[idx]];
            idx++;
            return tmp;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
}
