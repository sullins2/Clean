from GridWorld.Grid import *
from GridWorld.lonrAgent import *

import numpy as np


# Set amount to train
trainingIterations = 20

newGrid = Grid()

agent = Agent(gridworld=newGrid)

#print(newGrid.grid)

agent.train(iterations=trainingIterations)


print(agent.Q[36])

print("Done")