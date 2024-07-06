import cvxpy as cp
import numpy as np

# single author isotonic regression
def isotonic(y, ranking, lbd=None, mode='l2'):
	n = len(y)
	r = cp.Variable(n)

	objective = cp.Minimize( cp.norm2(y - r) ) 
	constraints = []
	for i in range( len(ranking)-1):
		constraints.append( r[ ranking[i] ] >= r[ ranking[i+1] ] )

	prob = cp.Problem(objective, constraints)
	result = prob.solve()
	return r.value

def multiisotonic(y, rankings, lbd=None, mode='l2'):
	score = np.zeros(len(y))
	for i in range(len(rankings)):
		score += isotonic(y, rankings[i], lbd, mode)
	score /= len(rankings)
	return score

