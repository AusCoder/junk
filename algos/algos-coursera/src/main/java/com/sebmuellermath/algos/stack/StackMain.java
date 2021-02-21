package com.sebmuellermath.algos.stack;

import java.util.stream.Stream;

public class StackMain {
  public static void main(String[] args) {
    Stack<String> stack = new Stack<>();
    Stream<String> ss = Stream.of(
      new String("to"),
      new String("be"),
      new String("or"),
      new String("not"),
      new String("to"),
      new String("-"),
      new String("be"),
      new String("-"),
      new String("-"),
      new String("that"),
      new String("-"),
      new String("-"),
      new String("-"),
      new String("is")
    );

    // ss.forEach(s -> {
    //   if (s.equals("-")) {
    //     System.out.println(stack.pop());
    //   } else {
    //     stack.push(s);
    //   }
    // });

    ss.forEach(s -> stack.push(s));
    for (String s : stack) {
      System.out.println(s);
    }
  }
}
