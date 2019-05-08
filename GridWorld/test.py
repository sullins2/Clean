from GridWorld.Grid import *
from GridWorld.lonrAgent import *

import numpy as np

#TODO
# Set start state
# Online version with all rewards revealed
# Last action into goal state doesn't currently have a living cost


# Set amount to train
trainingIterations = 22

newGrid = Grid()

agent = Agent(gridworld=newGrid)

#print(newGrid.grid)

agent.train(iterations=trainingIterations)

#Set start state

# 0 1 2
# 3 4 5

print(agent.Q[0])
print(agent.Q[1])
print(agent.Q[2])
print(agent.Q[3])
print(agent.Q[4])
print(agent.Q[5])



print("Done")