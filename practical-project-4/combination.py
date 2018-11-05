# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:48:50 2018

@author: FelipeLaranjeira
"""
import itertools

def partitions(n):
    assert isinstance(n, int), 'n must be an integer'
    assert n > 0, 'n must be a natural number but zero.'
    
    a = list(range(n+1))
    tuples = []
    
    for m in range(2, n+1):
        a[m] = 1
    
    m = 1
    a[0] = 0
    
    while(True):
        
        a[m] = n
        q = m - int(n == 1)
        
        while(True):
            tuples.append(tuple(a[1:m+1]))
            
            if(a[q] != 2):
                break
            
            a[q] = 1
            q -= 1
            m += 1
      
        if(q == 0):
            break
        
        x = a[q] - 1
        a[q] = x
        n = m - q + 1
        m = q + 1
        
        while(n > x):
            
            a[m] = x
            m += 1
            n -= x

    for subset in tuples:
        if not subset[1:] == subset[:-1]:
            new_subsets = list(itertools.permutations(subset))
            
            for ns in new_subsets:
                if ns not in tuples:
                    tuples.append(ns)
            
    return tuples
    
#tuples = partitions(4)
#print(tuples)