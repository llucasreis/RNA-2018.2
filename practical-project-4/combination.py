# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:48:50 2018

@author: FelipeLaranjeira
"""
import itertools

def partitions(n, condition_to_insert = None):
	"""Return a list with all integer partition tuples of n.

		Args:
			n (int): an integer to partition.
			condition_to_insert (:obj:`function`, optional): Function
				that controls if a tuple should enter the list. It should
				take one argument: a partition tuple of `n`. Defaults
				to None. If it is None, then every generated tuple will
				be inserted to the list.

		Returns:
			A list of all partition tuples that were accepted by
			`condition_to_insert` function.
	"""

	assert isinstance(n, int), 'n must be an integer'
	assert n > 0, 'n must be a natural number but zero.'

	if(condition_to_insert is None):
		condition_to_insert = lambda partition: True
	
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
			partition = tuple(a[1:m+1])

			if(condition_to_insert(partition)):
				permutations = list(set(list(itertools.permutations(partition))))
				tuples += permutations
						
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
			
	return tuples

#tuples = partitions(40, lambda partition: len(partition) <= 4)
#print(len(tuples))