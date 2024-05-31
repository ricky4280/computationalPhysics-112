import numpy as np

def grid_N(n, cutoff=True):
	grid = {}
	N = n**2
	for i in range(1,N+1):
		neb = []
		if not cutoff:
			if (i-6) > 0 and (i-6)%n !=0: neb.append(i-6)
			if (i-5) > 0: neb.append(i-5)
			if (i-4) > 0 and (i-4)%n !=1: neb.append(i-4)	
			if (i-1)%n != 0: neb.append(i-1)	
		neb.append(i)
		if (i+1)%n != 1: neb.append(i+1)
		if (i+4) < N and (i+4)%n != 0: neb.append(i+4)
		if (i+5) < N: neb.append(i+5)
		if (i+6) < N and (i+6)%n != 1: neb.append(i+6)
		grid[i] = np.array(neb)
	return grid
