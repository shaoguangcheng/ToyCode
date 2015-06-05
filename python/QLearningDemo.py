#!/usr/bin/env python
"""A very naive demo to show Q-Learning in path finding.(http://blog.csdn.net/pi9nc/article/details/27649323)"""

import random

R = [[-1, -1, -1, -1, 0, -1],
	[-1, -1, -1, 0, -1, 100],
	[-1, -1, -1, 0, -1, -1],
	[-1, 0, 0, -1, 0, -1],
	[0, -1, -1, 0, -1, 100],
	[-1, 0, -1, -1, 0, 100]];
target = 5;
nState = len(R);
nAction = len(R[0]);

def max_(x) :
	index = x.index(max(x))
	return (index, x[index])

def init() :
	Q = [[0]]*nState
	Q = [Q[x]*nAction for x in range(nState)]

	N = [];
	[N.append([]) for x in range(nState)]
	[N[x].append(y) for x in range(nState) for y in range(nAction) if R[x][y] >= 0]

	return (Q, N)

def learning(nEpisode, gamma) :
	(Q, N) = init()
	episode = 0

	while episode < nEpisode :
		state = random.randint(0, nState-1)
		episode = episode + 1	
		while True :
			n = len(N[state])
			stateNext = N[state][random.randint(0, n-1)]
			action = stateNext
			Q[state][action] = R[state][action] + gamma * max(Q[stateNext])
			state = stateNext
			if state == target :
				break;
	return Q

def run(startPoint, Q) :
	"input a new start point, return the best path to target"
	path = [startPoint];
	while startPoint != target :
		startPoint = max_(Q[startPoint])[0]
		path.append(startPoint)
	return path

if __name__ == "__main__" :
	Q = learning(1000, 0.8);
	print [run(x, Q) for x in range(nState)]
