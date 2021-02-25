package com.sebmuellermath.algos.deque;

import java.util.Iterator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class DequeTest {
    @Test
    public void testIterator() {
        Deque<Integer> d = new Deque<>();
        d.addLast(1);
        d.addLast(2);
        Iterator<Integer> it = d.iterator();
        Assertions.assertTrue(it.next().equals(1));
        Assertions.assertTrue(it.next().equals(2));
        Assertions.assertFalse(it.hasNext());
    }

    @Test
    public void testAddRemoveAdd_First() {
        Deque<Integer> d = new Deque<>();
        d.addFirst(1);
        d.removeLast();
        d.addFirst(1);
        Assertions.assertTrue(d.size() == 1);
    }

    @Test
    public void testAddRemoveAdd_Last() {
        Deque<Integer> d = new Deque<>();
        d.addLast(1);
        d.removeFirst();
        d.addLast(1);
        Assertions.assertTrue(d.size() == 1);
    }

    @Test
    public void testAddRemoveTwice() {
        Deque<Integer> d = new Deque<>();
        d.addFirst(3);
        d.addFirst(7);
        Assertions.assertTrue(d.size() == 2);
        Assertions.assertTrue(d.removeLast().equals(3));
        Assertions.assertTrue(d.size() == 1);
        Assertions.assertTrue(d.removeLast().equals(7));
        Assertions.assertTrue(d.size() == 0);
    }

    @Test
    public void testAddFirstRemoveLastSize() {
        Deque<Integer> d = new Deque<>();
        d.addFirst(3);
        d.removeLast();
        Assertions.assertTrue(d.size() == 0);
    }

    @Test
    public void testAddFirstRemoveLast() {
        Deque<Integer> d = new Deque<>();
        d.addFirst(3);
        Assertions.assertTrue(d.removeLast().equals(3));
    }

    @Test
    public void testAddFirstSize() {
        Deque<Integer> d = new Deque<>();
        d.addFirst(1);
        Assertions.assertTrue(d.size() == 1);
    }

    @Test
    public void testEmptySize() {
        Deque<Integer> d = new Deque<>();
        Assertions.assertTrue(d.size() == 0);
    }
}
