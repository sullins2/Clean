from LONRGeneral.LONR import *
from LONRGeneral.GridWorld import *
from LONRGeneral.Soccer import *
from LONRGeneral.NoSDE import *
from LONRGeneral.TigerGame import *
from LONRGeneral.TwoState import *
from LONRGeneral.AbsentDriver import *



# TODO: CLEAN UP
#   STANDARDIZE EVERYTHING (GridWorld into other forms)
#   Check and see the best way to standardize
#   Add LONR-V for Tiger Game

####################################################
# ABSENT MINDED DRIVER
####################################################
# absDriver = AbsentDriver(startState=0)
# gamma = 1.0
# parameters = {'alpha': 0.99, 'epsilon': 30, 'gamma': gamma}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
# lonrAgent = LONR_V(M=absDriver, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters= 140000
# lonrAgent.lonr_train(iterations=iters, log=10000)
#
# lonrAgent.printOut()



####################################################
# TWO STATE MDP
####################################################
# twoState = TwoState()
# gamma = 0.99
#
# parameters = {'alpha': 1.0, 'epsilon': None, 'gamma': gamma}
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True} # DCFR parameters alpha, beta, gamma?
#
# lonrAgent = LONR_V(M=twoState, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters= 20000
# lonrAgent.lonr_train(iterations=iters, log=12250)
#
# lonrAgent.printOut()
#
# print("J(1): ", (2.0 * gamma)/ (3.0 * (1.0 - gamma)))
# print("J(2): ", 1.0 + (2.0 * gamma)/ (3.0 * (1.0 - gamma)))




####################################################
# GridWorld MDP - V
####################################################
# gridMDP = Grid(noise=0.0)
#
# parameters = {'alpha': 1.0, 'epsilon': None, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
#
# lonrAgent = LONR_V(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters=1000
# lonrAgent.lonr_train(iterations=iters, log=1250)
#
# lonrAgent.printOut()


####################################################
# GridWorld MDP - A
####################################################

# # Create GridWorld (inherits from MDP class)
# print("Begin LONR-A tests on GridWorld")
#
# gridMDP = Grid(noise=0.0, startState=36)
#
# # Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# lonrAgent = LONR_A(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
#
# # Train via VI
# iters = 2000
# lonrAgent.lonr_train(iterations=iters, log=500)
#
# lonrAgent.printOut()



####################################################
# NoSDE - V
####################################################
# print("SET GAMMA TO 0.75")
# noSDE = NoSDE()
#
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 0.75}
#
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
#
# iters = 140000
# lonrAgent.lonr_train(iterations=iters, log=5000)
#
# lonrAgent.printOut()


#######################################################
# Tiger Game AVI
#######################################################

tigerGame = TigerGame(startState="START", TLProb=0.5, version=1)

# Create LONR Agent and feed in the Tiger game
parameters = {'alpha': 0.99, 'epsilon': 20, 'gamma': 0.99}
regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}

lonrAgent = LONR_V(M=tigerGame, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})

iters = 50000
lonrAgent.lonr_train(iterations=iters, log=2500, randomize=True)

lonrAgent.printOut()




#######################################################
#  NoSDE LONR-B
#######################################################
# noSDE = NoSDE()
#
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 0.75}
#
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_B(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
#
# iters = 10000
# lonrAgent.lonr_train(iterations=iters, log=10000)
#
# lonrAgent.printOut()


#######################################################
# GridWorld LONR-B
#######################################################
# iters = 3000
# gridMDP = Grid(startState=8)
# parameters = {'alpha': 0.99, 'epsilon': 20, 'gamma': 0.99}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# lonrAgent = LONR_B(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers) #JUST SWITCH BACK TO B
# lonrAgent.lonr_train(iterations=iters, log=100)
# lonrAgent.printOut()



###################################################################
        # # Q learning
        # print("Begin GridWorld VI with non-determinism")
        # gridMDP = Grid(noise=0.20, startState=36)
        #
        # # Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
        # parameters = {'alpha': 0.4, 'epsilon': 40, 'gamma': 1.0, 'alphaDecay' : 1.0}
        # regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
        # lonrAgent = LONR_A3(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
        #
        # # Train via VI
        # iters = 50000
        # lonrAgent.qlearning_train(iterations=iters, log=3500)
        #
        # print("[North, East, South, West]")
        # print("Note: these are Q, not QAvg")
        # print("Q for: Bottom left (start state) (West opt = 170.23]")
        # print(gridMDP.Q[0][36])
        #
        # print("Q for: State above bottom right terminal")
        # print(gridMDP.Q[0][35])
        # print("End GridWorld VI with non-determism")
        # print("")
        # print("")
        # print("Q")
        # for k in sorted(gridMDP.Q[0].keys()):
        #     tot = 0.0
        #     print(k, ": ", end='')
        #     maxA = -10000
        #     maxQ = -10000
        #     for kk in gridMDP.Q[0][k].keys():
        #         if gridMDP.Q[0][k][kk] > maxQ:
        #             maxQ = gridMDP.Q[0][k][kk]
        #             maxA = kk
        #     print(": BEST ACTION: ", maxA, "  ", end='')
        #     for kk in gridMDP.Q[0][k].keys():
        #
        #         print(kk, " : ", gridMDP.Q[0][k][kk], " ", end='')
        #
        #
        #     print("")
        #
        # print("")
        # gridMDP.printGrid()








print("")
print("Done")




















