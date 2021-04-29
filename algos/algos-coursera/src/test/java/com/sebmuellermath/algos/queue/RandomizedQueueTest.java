package com.sebmuellermath.algos.queue;

import java.util.Iterator;
import java.util.HashSet;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class RandomizedQueueTest {
    @Test
    public void testEnqueueResize() {
        RandomQueue<Integer> q = new RandomQueue<>();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        Assertions.assertTrue(q.size() == 3);
    }

    @Test
    public void testDequeue() {
        RandomQueue<Integer> q = new RandomQueue<>();
        HashSet<Integer> s = new HashSet<>();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        for (int i = 0; i < 3; i++) {
            s.add(q.dequeue());
        }
        Assertions.assertTrue(s.size() == 3);
        Assertions.assertTrue(s.contains(1));
        Assertions.assertTrue(s.contains(2));
        Assertions.assertTrue(s.contains(3));
    }

    @Test
    public void testIterator() {
        RandomQueue<Integer> q = new RandomQueue<>();
        HashSet<Integer> s = new HashSet<>();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        for (Integer item : q) {
            s.add(item);
        }
        Assertions.assertTrue(s.size() == 3);
        Assertions.assertTrue(s.contains(1));
        Assertions.assertTrue(s.contains(2));
        Assertions.assertTrue(s.contains(3));
    }
}
