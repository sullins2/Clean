from GridWorld.Grid import *
from GridWorld.lonrAgent import *

import numpy as np

#TODO
# Set start state
# Online version with all rewards revealed
# Last action into goal state doesn't currently have a living cost


# Set amount to train
trainingIterations = 3201

newGrid = Grid()

agent = Agent(gridworld=newGrid)

print(newGrid.grid)

# agent.train(iterations=trainingIterations)

agent.train_olonr(totalIterations=trainingIterations)

#
#print("Non-Deterministic O-LONR check avg Q also add in a way to run it with actions non deterministic")

#Set start state

# 0 1 2
# 3 4 5

# print(agent.Q[0])
# print(agent.Q[1])
# print(agent.Q[2])
# print(agent.Q[3])
# print(agent.Q[4])
# print(agent.Q[5])

# print("")
# print("Online:")
#
# agent = Agent(gridworld=newGrid)
#
# totalEpisodes = 2222
# agent.train_olonr(totalIterations=totalEpisodes)
# for ep in range(totalEpisodes):
#     agent.lonr_episode()

#
# print(agent.Q[0])
# print(agent.Q[1])
# print(agent.Q[2])
# print(agent.Q[3])
# print(agent.Q[4])
# print(agent.Q[5])
#
# print("Pi sums")
# print(agent.pi_sums[0])
# print(agent.pi_sums[1])
# print(agent.pi_sums[2])
# print(agent.pi_sums[3])
# print(agent.pi_sums[4])
# print(agent.pi_sums[5])
#
# print("Pi")
# print(agent.pi[0])
# print(agent.pi[1])
# print(agent.pi[2])
# print(agent.pi[3])
# print(agent.pi[4])
# print(agent.pi[5])
#
# print("Pi 35 THIS IS ABOVE GOAL")
# print(agent.pi[35])
# print("Pi_sums 35")
# print(agent.pi_sums[35])
#
#
# print("Pi 36 THIS IS BOTTOM LEFT")
# print(agent.pi[36])
# print("Pi_sums 36")
# print(agent.pi_sums[36])
#
# print("Pi 24 THIS IS BOTTOM LEFT")
# print(agent.pi[24])
# print("Pi_sums 24")
# print(agent.pi_sums[24])
#
# print("Q:")
# print(agent.Q[36])

print("36: ", agent.pi[36])
print("35", agent.pi[35])
print("Q36", agent.Q[36])
print("Q35", agent.Q[35])


print("Done")