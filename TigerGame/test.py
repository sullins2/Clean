
from TigerGame.Tiger import *

OPENLEFT = "OL"
LISTEN = "L"
OPENRIGHT = "OR"
GL = "GL"
GR = "GR"
totStates = []
QValues = {}
QValues_backup = {}
pi = {}
regret_sums = {}
pi_sums = {}

rewardSums = []


if __name__ == '__main__':


    trainingIterations = 14444

    newTigerGame = TigerGame()

    agent = TigerAgent(tigergame=newTigerGame)

    agent.train_lonr_value_iteration(iterations=trainingIterations)

    print("")
    print("")
    print("Agent pi:")
    for k in sorted(agent.pi.keys()):
        if len(list(agent.pi[k].keys())) > 1:
            print(k, ":  ", agent.pi[k])
    print("")
    print("Agent pi_sums:")
    for k in sorted(agent.pi_sums.keys()):
        if len(list(agent.pi_sums[k].keys())) > 1:
            print(k, ":  ", agent.pi_sums[k])
    print("")
    print("Agent Q:")
    for k in sorted(agent.Q.keys()):
        if len(list(agent.Q[k].keys())) > 1:
            print(k, ":  ", agent.Q[k])
    print("")
    print("Agent regret sums:")
    for k in sorted(agent.regret_sums.keys()):
        if len(list(agent.regret_sums[k].keys())) > 1:
            print(k, ":  ", agent.regret_sums[k])
