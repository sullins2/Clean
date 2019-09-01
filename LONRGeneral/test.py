from LONRGeneral.LONR import *
from LONRGeneral.GridWorld import *
from LONRGeneral.Soccer import *
from LONRGeneral.NoSDE import *
from LONRGeneral.TigerGame import *
from LONRGeneral.TwoState import *
from LONRGeneral.AbsentDriver import *
from cvxopt import *
from LONRGeneral.misc import *
from LONRGeneral.Kuhn import *
from MarkovGames.soccer import *
from LONRGeneral.ShapleysGame import *
from LONRGeneral.BiasedGame import *
from LONRGeneral.TrickyGame import *
from LONRGeneral.Prisoner import *
from LONRGeneral.RPS import *
from LONRGeneral.MatchingPennies import *


# TODO: CLEAN UP
#   STANDARDIZE EVERYTHING (GridWorld into other forms)
#   Check and see the best way to standardize
#   Add LONR-V for Tiger Game


####################################################
## Soccer LONR_AB
###################################################
# soccer = SoccerGame()
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.9}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=soccer, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
#
# iters = 20000
# lonrAgent.lonr_train(iterations=iters, log=1000)
#
# lonrAgent.printOut()
#
# soccer.normalize_pisums()
# print("")
# print("PiA21 A: ", lonrAgent.M.pi_sums[0]["A21"])
# print("PiA21 B: ", lonrAgent.M.pi_sums[1]["A21"])
# print("")
# print("PiA25 A: ", lonrAgent.M.pi_sums[0]["A25"])
# print("PiA25 B: ", lonrAgent.M.pi_sums[1]["A25"])
# print("")
# print("PiA61 A: ", lonrAgent.M.pi_sums[0]["A61"])
# print("PiA61 B: ", lonrAgent.M.pi_sums[1]["A61"])
# print("")
#
# print("PiA15 A: ", lonrAgent.M.pi_sums[0]["A15"])
# print("PiA15 B: ", lonrAgent.M.pi_sums[1]["A15"])
# print("")
# print("PiB61 A: ", lonrAgent.M.pi_sums[0]["B61"])
# print("PiB61 B: ", lonrAgent.M.pi_sums[1]["B61"])
# print("")
# print("PiB25 A: ", lonrAgent.M.pi_sums[0]["B25"])
# print("PiB25 B: ", lonrAgent.M.pi_sums[1]["B25"])
# print("")
# print("PiA10 A: ", lonrAgent.M.pi_sums[0]["A10"])
# print("PiA10 B: ", lonrAgent.M.pi_sums[1]["A10"])
# print("")
# print("PiA14 A: ", lonrAgent.M.pi_sums[0]["A14"])
# print("PiA14 B: ", lonrAgent.M.pi_sums[1]["A14"])
# print("")
#
#
# print("Playing random:")
# soccer.play(iterations=20000, log=5000)
# print("Done playing random")
# print("")
# print("Random Games")
# soccer.play_random(iterations=20000, log=5000)

# print("")
# print("Random Games and randomMoves")
# soccer.play_random_player(iterations=30000, log=23000)


######################################################
# Matching Pennies LONR_V
######################################################
# sg = MatchingPennies()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.99, 'epsilon': 30, 'gamma': 0.99}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# # dcfr = {'alphaDCFR' : math.inf, 'betaDCFR' : -math.inf, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=sg, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 100000
# lonrAgent.lonr_train(iterations=iters, log=7140)
#
# lonrAgent.printOut()


#####################################################
# RPS LONR_V
#####################################################
# sg = RPSGame()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# # dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_AB(M=sg, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 30000
# lonrAgent.lonr_train(iterations=iters, log=7140)
#
# lonrAgent.printOut()


######################################################
# Prisoner LONR_V
######################################################
# sg = Prisoner()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.199, 'epsilon': 30, 'gamma': 0.99}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=sg, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 150
# lonrAgent.lonr_train(iterations=iters, log=7140)
#
# lonrAgent.printOut()


######################################################
# Tricky Game LONR_V
######################################################
# sg = TrickyGame()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=sg, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 2500
# lonrAgent.lonr_train(iterations=iters, log=31140)
#
# lonrAgent.printOut()



######################################################
# Shapley's Game LONR_V
######################################################
# sg = ShapleysGame()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
#
# # regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=sg, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 2500
# lonrAgent.lonr_train(iterations=iters, log=31140)
#
# lonrAgent.printOut()



######################################################
# Biased Game LONR_V
######################################################
# sg = BiasedGame()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=sg, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 21251
# lonrAgent.lonr_train(iterations=iters, log=31140)
#
# lonrAgent.printOut()


####################################################
# RPS LONR-V
####################################################
noSDE = RPSGame()
print("Make sure to put > 2 in, no terminal")
# lvc=LastValueCount, hv=HighestValue
parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99, 'lvc': 3.0, 'hv': 1.0, 'eta': 1.0/32.0, 'maxGap': 0.5}

regret_minimizers = {'RM': True, 'RMPlus': False, 'RMPlusPlus': False, 'DCFR': False, 'MWU': False, 'OMWU': False}
# regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}

dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
lonrAgent = LONR_V(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
iters = 5
lonrAgent.lonr_train(iterations=iters, log=14000)

lonrAgent.printOut()

print(lonrAgent.piTRAJ)

####################################################
# NOSDE LONR-V
####################################################
# noSDE = NoSDE()
# print("Make sure to put > 2 in, no terminal")
# # lvc=LastValueCount, hv=HighestValue
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.75, 'lvc': 9.0, 'hv': 12.0, "eta": 1.0/2.0}
#
# regret_minimizers = {'RM': False, 'RMPlus': False, 'RNPlusPlus': False, 'DCFR': False, 'MWU': True, 'OMWU': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
#
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_V(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 50000
# lonrAgent.lonr_train(iterations=iters, log=4000)
#
# lonrAgent.printOut()




####################################################
# NOSDE LONR-AA
####################################################
# noSDE = NoSDE()
# print("Make sure to put > 2 in, no terminal")
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.75}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # regret_minimizers = {'RM': False, 'RMPlus': True, 'DCFR': False}
# # regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_AB(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
# iters = 20000
# lonrAgent.lonr_train(iterations=iters, log=4000)
#
# lonrAgent.printOut()



####################################################
# ABSENT MINDED DRIVER
####################################################
# absDriver = AbsentDriver(startState=0)
# gamma = 0.99
# parameters = {'alpha': 0.199, 'epsilon': 10, 'gamma': gamma}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
# lonrAgent = LONR_V(M=absDriver, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters= 55000
# lonrAgent.lonr_train(iterations=iters, log=10000, randomize=True)
#
# lonrAgent.printOut()



####################################################
# TWO STATE MDP
####################################################
# twoState = TwoState()
# gamma = 0.99
#
# parameters = {'alpha': 1.0, 'epsilon': None, 'gamma': gamma}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
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
# gridMDP = Grid(noise=0.0, startState=36)
#
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
#
# lonrAgent = LONR_V(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters=1130
# lonrAgent.lonr_train(iterations=iters, log=325)
#
# lonrAgent.printOut()


####################################################
# GridWorld MDP - A
####################################################

# Create GridWorld (inherits from MDP class)
# print("Begin LONR-A tests on GridWorld")
#
# gridMDP = Grid(noise=0.0, startState=36)
#
# # Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# lonrAgent = LONR_AV(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
#
# # Train via VI
# iters = 120000
# lonrAgent.lonr_train(iterations=iters, log=2500)
#
# lonrAgent.printOut()



####################################################
# NoSDE - V
####################################################
# print("SET GAMMA TO 0.75")
# noSDE = NoSDE()
#
# parameters = {'alpha': 1.0, 'epsilon': 10, 'gamma': 0.75}
#
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_AAA(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
#
# iters = 40000
# lonrAgent.lonr_train(iterations=iters, log=5000)
#
# lonrAgent.printOut()



####################################################
# Correlated Equilibrium - Linear Program
####################################################
# A = [[0, (3.0*0.75)/ 0.4375], [9, 4], [4, 0], [4, 4]]
# def ce(A, solver=None):
#     print("")
#     print("Correlated equilibrium")
#
#     num_vars = len(A)
#     print("num_vars: ", num_vars)
#     # maximize matrix c
#     c = [sum(i) for i in A] # sum of payoffs for both players
#     print("c: ", c)
#     c = np.array(c, dtype="float")
#     c = matrix(c)
#     c *= -1 # cvxopt minimizes so *-1 to maximize
#     print("c *-1: ", c)
#     # constraints G*x <= h
#     G = build_ce_constraints(A=A)
#     print("G: ")
#     print(G)
#     G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
#
#     h_size = len(G)
#     G = matrix(G)
#     h = [0 for i in range(h_size)]
#     h = np.array(h, dtype="float")
#     h = matrix(h)
#     # contraints Ax = b
#     A = [1 for i in range(num_vars)]
#     A = np.matrix(A, dtype="float")
#     A = matrix(A)
#     b = np.matrix(1, dtype="float")
#     b = matrix(b)
#     print("")
#     sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
#     return sol
#
# def build_ce_constraints(A):
#     num_vars = int(len(A) ** (1/2))
#     print("num_vars ** (1/2): ", num_vars)
#     G = []
#     # row player
#     for i in range(num_vars): # action row i
#         for j in range(num_vars): # action row j
#             if i != j:
#                 constraints = [0 for i in A]
#                 base_idx = i * num_vars
#                 comp_idx = j * num_vars
#                 for k in range(num_vars):
#                     constraints[base_idx+k] = (- A[base_idx+k][0] + A[comp_idx+k][0])
#                 G += [constraints]
#     print("Row player G: ")
#     print(G)
#     # col player
#     for i in range(num_vars): # action column i
#         for j in range(num_vars): # action column j
#             if i != j:
#                 constraints = [0 for i in A]
#                 for k in range(num_vars):
#                     constraints[i + (k * num_vars)] = (
#                         - A[i + (k * num_vars)][1]
#                         + A[j + (k * num_vars)][1])
#                 G += [constraints]
#     return np.matrix(G, dtype="float")
#
# sol = ce(A=A, solver="glpk")
# probs = sol["x"]
# print("")
# print(probs)





#######################################################
# Tiger Game A
#######################################################

# tigerGame = TigerGame(startState="START", TLProb=0.5, version=1)
#
# # Create LONR Agent and feed in the Tiger game
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
#
# lonrAgent = LONR_A(M=tigerGame, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters = 30000
# lonrAgent.lonr_train(iterations=iters, log=2500, randomize=True)
#
# lonrAgent.printOut()

#######################################################
# Tiger Game AVI
#######################################################

# tigerGame = TigerGame(startState="START", TLProb=0.5, version=1)
#
# # Create LONR Agent and feed in the Tiger game
# parameters = {'alpha': 0.199, 'epsilon': 10, 'gamma': 0.99}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
#
# lonrAgent = LONR_AV(M=tigerGame, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters = 50000
# lonrAgent.lonr_train(iterations=iters, log=12500, randomize=True)
#
# lonrAgent.printOut()




#######################################################
#  NoSDE LONR-B
# TODO: LONR-B for n>1
#######################################################
# noSDE = NoSDE()
#
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 0.75}
#
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# dcfr = {'alphaDCFR' : 3.0 / 2.0, 'betaDCFR' : 0.0, 'gammaDCFR' : 2.0}
# lonrAgent = LONR_A(M=noSDE, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
#
# iters = 10000
# lonrAgent.lonr_train(iterations=iters, log=10000)
#
# lonrAgent.printOut()


#######################################################
# GridWorld LONR-B
#######################################################
# iters = 40000
# gridMDP = Grid(startState=36)
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# lonrAgent = LONR_B(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
# lonrAgent.lonr_train(iterations=iters, log=1500)
# lonrAgent.printOut()


#######################################################
# GridWorld LONR-TD
#######################################################
# print("HERE")
# iters = 15000
# gridMDP = Grid(startState=36)#168 for big one
# parameters = {'alpha': 0.01, 'epsilon': 10, 'gamma': 0.99, 'alphaDecay': 1.0}
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True}
# lonrAgent = LONR_TD(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
# lonrAgent.lonr_train(iterations=iters, log=1511)
# lonrAgent.printOut()

# #######################################################
# # Tiger Game LONR-TD
# #######################################################
#
# tigerGame = TigerGame(startState="START", TLProb=0.5, version=1)
#
# # Create LONR Agent and feed in the Tiger game
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
#
# lonrAgent = LONR_TD(M=tigerGame, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters = 160000
# lonrAgent.lonr_train(iterations=iters, log=2500, randomize=True)
#
# lonrAgent.printOut()


#######################################################
# Tiger Game Q-LEARNING
#######################################################

# tigerGame = TigerGame(startState="START", TLProb=0.5, version=1)
#
# # Create LONR Agent and feed in the Tiger game
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
#
# lonrAgent = QLEARN(M=tigerGame, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters = 422000
# lonrAgent.lonr_train(iterations=iters, log=2500, randomize=True)
#
# lonrAgent.printOut()

####################################################
# ABSENT MINDED DRIVER - LONR-TD
####################################################
# absDriver = AbsentDriver(startState=0)
# gamma = 1.0
# parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': gamma}
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True} # DCFR parameters alpha, beta, gamma?
# lonrAgent = LONR_TD(M=absDriver, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters= 150000
# lonrAgent.lonr_train(iterations=iters, log=12100, randomize=True)
#
# lonrAgent.printOut()


####################################################
# ABSENT MINDED DRIVER - LONR-B
####################################################
# absDriver = AbsentDriver(startState=0)
# gamma = 1.0
# parameters = {'alpha': 0.99, 'epsilon': 31, 'gamma': gamma}
# regret_minimizers = {'RM': False, 'RMPlus': False, 'DCFR': True} # DCFR parameters alpha, beta, gamma?
# lonrAgent = LONR_B(M=absDriver, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
#
# iters= 75000
# lonrAgent.lonr_train(iterations=iters, log=2100, randomize=True)
#
# lonrAgent.printOut()

###########################################################
# Kuhn Poker
###########################################################
# print("Kuhn Poker:")
#
# #This fills totState and totInfoStates with states, (Creates MDP)
# root = KuhnRootChanceGameState(actions=deck)
#
# lonr_agent = Agent(totState=totState, infosets=infosets, totInfoStates=totInfoStates)
#
# LEPS = 50000
#
# train_lonr(lonr_agent, n_episodes=LEPS, T=LEPS)
#
# print("NE BELOW:")
# lonr_agent.compute_nash_equilibrium()
# for k in sorted(lonr_agent.nash_equilibrium):
#     print(k, "   ", lonr_agent.nash_equilibrium[k])
# print('')
# print(lonr_agent.nash_equilibrium)
# print('---------------------------------')
# print("GaV: ", lonr_agent.game_value())
#
# # kuhn = KuhnRootChanceGameState(actions=deck)
# # kuhnMDP = KuhnMDP()
# #
# # parameters = {'alpha': 0.99, 'epsilon': 10, 'gamma': 0.99}
# # regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# #
# # # lonrAgent = LONR_VCF(M=kuhn, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})
# #
# # for s in kuhnMDP.getStates():
# #     print(kuhnMDP.getStateRep(s))
#############################################################




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




















