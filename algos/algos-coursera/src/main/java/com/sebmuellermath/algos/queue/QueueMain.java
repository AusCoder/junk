package com.sebmuellermath.algos.queue;

import java.util.stream.Stream;

public class QueueMain {
  public static void main(String[] args) {
    Queue<String> queue = new Queue<>();
    Stream<String> ss = Stream.of(
      new String("a"),
      new String("b")
      // new String("to")
      // new String("be")
    );

    ss.forEach(s -> queue.enqueue(s));
    // for (String s : queue) {
    //   System.out.println(s);
    // }
    System.out.println(queue.dequeue());
    System.out.println(queue.dequeue());
    // System.out.println(queue.dequeue());
    // System.out.println(queue.dequeue());
  }
}
