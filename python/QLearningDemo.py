#!/usr/bin/env python
"""A very naive demo to show Q-Learning in path finding."""

import random

# Some global settings
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
	index = 0
	for i in range(1, len(x)) :
		if x[index] < x[i] : 
			index = i
	return (index, x[index])

def init() :
	"init Q and N"	
	Q = [[0]]*nState
	Q = [Q[x]*nAction for x in range(nState)]

	N = [];
	[N.append([]) for x in range(nState)]
	[N[x].append(y) for x in range(nState) for y in range(nAction) if R[x][y] >= 0]

	return (Q, N)

def learning(nEpisode, gamma) :
	"learning Q matrix"
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
	"input a new startPoint, return the best path to target"
	path = [startPoint];
	while startPoint != target :
		startPoint = max_(Q[startPoint])[0]
		path.append(startPoint)
	return path

if __name__ == "__main__" :
	Q = learning(1000, 0.8);
	for x in range(nState) :
		print run(x, Q)
