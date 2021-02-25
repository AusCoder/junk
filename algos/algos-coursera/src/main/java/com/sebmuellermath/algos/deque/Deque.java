package com.sebmuellermath.algos.deque;

import java.util.Iterator;
import java.util.NoSuchElementException;


public class Deque<A> implements Iterable<A> {
    private Node head;
    private Node tail;

    public Deque() {
        head = null;
        tail = null;
    }

    public int isEmpty() {
        return head == null;
    }

    public int size() {
        int s = 0;
        Node t = tail;
        while (t != null) {
            s++;
            t = t.next;
        }
        return s;
    }

    public void addFirst(A item) {
        validateItem(item);
        Node newNode = new Node(item, null, head);
        if (head == null) {
            assert(tail == null);
            head = newNode;
            tail = newNode;
        } else {
            head.next = newNode;
            head = newNode;
        }
    }

    public void addLast(A item) {
        validateItem(item);
        Node newNode = new Node(item, tail, null);
        if (tail == null) {
            assert(head == null);
            head = newNode;
            tail = newNode;
        } else {
            tail.prev = newNode;
            tail = newNode;
        }
    }

    private void validateItem(A item) {
        if (item == null) {
            throw new IllegalArgumentException();
        }
    }

    public A removeFirst() {
        if (head == null) {
            assert(tail == null);
            throw new NoSuchElementException();
        } else {
            assert(tail != null);
            Node x = head;
            head = x.prev;
            if (head == null) {
                tail = null;
            } else {
                head.next = null;
            }
            x.next = null;
            x.prev = null;
            return x.item;
        }
    }

    public A removeLast() {
        if (tail == null) {
            assert(head == null);
            throw new NoSuchElementException();
        } else {
            assert(head != null);
            Node x = tail;
            tail = x.next;
            if (tail == null) {
                head = null;
            } else {
                tail.prev = null;
            }
            x.next = null;
            x.prev = null;
            return x.item;
        }
    }

    private class Node {
        public A item;
        public Node next;
        public Node prev;

        public Node(A i, Node n, Node p) {
            item = i;
            next = n;
            prev = p;
        }
    }

    @Override
    public Iterator<A> iterator() {
        return new DequeIterator();
    }

    private class DequeIterator implements Iterator<A> {
        private Node n = head;

        @Override
        public boolean hasNext() {
            return n != null;
        }

        @Override
        public A next() {
            if (n == null) {
                throw new NoSuchElementException();
            }
            A item = n.item;
            n = n.prev;
            return item;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
}
