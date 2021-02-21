package com.sebmuellermath.algos.unionfind;
/*
Social network. You have a sorted log file of form:

  [<timestamp>] X is friends with Y
  ...

Find the minimum time that everyone is connected to everyone
else via friends.
*/

import java.util.stream.Stream;

public class SocialNetworkConnected {
  private int size;
  private UnionFind unionFind;

  public SocialNetworkConnected(int n) {
    size = n;
    unionFind = new WeightedQuickUnion(n);
  }

  public int earliestTimeStamp(Stream<Log> logs) {
    int numGroups = size;
    // logs
    //   .mapToInt(log -> {
    //     if (!unionFind.isConnected(log.person1, log.person2)) {
    //       size--;
    //       unionFind.union(log.person1, log.person2);
    //     }
    //     return log.timestamp;
    //   })
    //   .dropWhile(ts -> size > 1);
    //   .

    // logs
    //   .takeWhile(log -> size > 1)
    //   .forEach(log -> {
    //     // System.out.println(size);
    //     // System.out.println(log.timestamp);
    //     if (!unionFind.isConnected(log.person1, log.person2)) {
    //       size--;
    //       unionFind.union(log.person1, log.person2);
    //     }
    //   });

    // logs.forEach(log -> {
    //   System.out.println(log.person1);
    // });
    return -1;
  }

  public static void main(String[] args) {
    SocialNetworkConnected network = new SocialNetworkConnected(4);
    Stream<Log> logs = Stream.of(
      new Log(0, 0, 1),
      new Log(1, 1, 2),
      new Log(2, 2, 0),
      new Log(3, 0, 1),
      new Log(4, 0, 3),
      new Log(5, 0, 3)
    );
    network.earliestTimeStamp(logs);
  }
}

class Log {
  public int timestamp;
  public int person1;
  public int person2;

  public Log(int t, int p1, int p2) {
    timestamp = t;
    person1 = p1;
    person2 = p2;
  }
}
