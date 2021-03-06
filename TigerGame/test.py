
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


    trainingIterations = 20000

    newTigerGame = TigerGame()

    agent = TigerAgent(tigergame=newTigerGame)

    agent.train_lonr_value_iteration(iterations=trainingIterations)

    #cl, cr = agent.train_lonr_online(iterations=trainingIterations, log=1000)

    # print(cl, "  ", cr)
    # print("")
    # print("")
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

    print("")
    print("Q Avg")
    for k in sorted(agent.Qsums.keys()):
        tot = 0.0
        # for kk in agent.Qsums[k].keys():
        #     tot += agent.Qsums[k][kk]
        # if tot == 0:
        #     tot = 1.0
        print(k, ": ", end='')
        for kk in agent.Qsums[k].keys():
            print(kk, ": ", agent.Qsums[k][kk] / (float(trainingIterations) / 2.0), " ", end='')
        print("")

    print("")
    ###################################
    # check this is correct
    # print("Agent Q sums:")
    # for k in sorted(agent.Qsums.keys()):
    #     if len(list(agent.Qsums[k].keys())) > 1:
    #         print(k, " ", end='')
    #         for kk in agent.Qsums[k].keys():
    #             print(kk, ":  ", agent.Qsums[k][kk] / float(trainingIterations), " ", end="")
    #
    #         print("")