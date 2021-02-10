package com.sebmuellermath.algos;

import com.sebmuellermath.algos.unionfind.QuickUnion;
import com.sebmuellermath.algos.unionfind.WeightedQuickUnion;

public class App
{
    public static void main( String[] args )
    {
        // QuickUnion uf = new QuickUnion(3);
        WeightedQuickUnion uf = new WeightedQuickUnion(3);
        uf.union(0,1);
        uf.union(1,2);
        System.out.println(uf.depth());
    }
}
