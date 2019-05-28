from LONRGeneral.LONR import *
from LONRGeneral.GridWorld import *
from LONRGeneral.Soccer import *
from LONRGeneral.NoSDE import *
from LONRGeneral.TigerGame import *

####################################################
# GridWorld MDP
####################################################

# Create GridWorld (inherits from MDP class)
# print("Begin LONR-V tests on GridWorld")
#
# print("Creating deterministic Grid World")
# gridMDP = Grid(noise=0.0)
#
# # Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
# parameters = {'alpha': 1.0, 'epsilon': None, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
#
# lonrAgent = LONR_V(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
#
# # Train via VI
# lonrAgent.lonr_train(iterations=550, log=150)
# print("[North, East, South, West]")
# print("Note: these are Q, not QAvg")
# print("Q for: Bottom left (start state)")
# print(gridMDP.Q[0][36])
#
# print("Q for: State above bottom right terminal")
# print(gridMDP.Q[0][35])
# print("End GridWorld VI with determism")

#########################################################################
# print("")
#
# # Create GridWorld (inherits from MDP class)
# print("Begin LONR-A tests on GridWorld")
#
# print("Creating deterministic Grid World")
# gridMDP = Grid(noise=0.20, startState=36)
#
# # Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False} # DCFR parameters alpha, beta, gamma?
#
# lonrAgent = LONR_A(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
#
# # Train via VI
# lonrAgent.lonr_train(iterations=1, log=250)
# print("[North, East, South, West]")
# print("Note: these are Q, not QAvg")
# print("Q for: Bottom left (start state)")
# print(gridMDP.Q[0][36])
#
# print("Q for: State above bottom right terminal")
# print(gridMDP.Q[0][35])
# print("End GridWorld VI with determism")


# # # # Create GridWorld (inherits from MDP class)


# All should be good with pi is good with epsilon=20, 13000 iterations
## 528 - these work with LONR_A and LONR_V, w/wo noise
# print("Begin GridWorld VI with non-determinism")
# gridMDP = Grid(noise=0.20, startState=36)
#
# # Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
# parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': True, 'DCFR': False}
# lonrAgent = LONR_A(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)
#
# # Train via VI
# iters = 2
# lonrAgent.lonr_train(iterations=iters, log=500)
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
# print("Q Avg")
# for k in sorted(gridMDP.QSums[0].keys()):
#     tot = 0.0
#     print(k, ": ", end='')
#     for kk in gridMDP.QSums[0][k].keys():
#         touched = gridMDP.QTouched[0][k][kk]
#         if touched == 0: touched = 1.0
#         print(kk, ": ", gridMDP.QSums[0][k][kk] / touched, " ", end='')
#     print("")
#
# print("")
# print("Pi Sums")
# for k in sorted(gridMDP.pi_sums[0].keys()):
#     tot = 0.0
#     for kk in gridMDP.pi_sums[0][k].keys():
#         tot += gridMDP.pi_sums[0][k][kk]
#     if tot == 0: tot = 1.0
#     print(k, ": ", end='')
#     for kk in gridMDP.pi_sums[0][k].keys():
#         print(kk, ": ", gridMDP.pi_sums[0][k][kk] / tot, " ", end='')
#     print("")
#
# print("")
# print("Pi")
# for k in sorted(gridMDP.pi[0].keys()):
#     tot = 0.0
#     for kk in gridMDP.pi[0][k].keys():
#         tot += gridMDP.pi[0][k][kk]
#     if tot == 0: tot = 1.0
#     print(k, ": ", end='')
#     for kk in gridMDP.pi[0][k].keys():
#         print(kk, ": ", gridMDP.pi[0][k][kk], " ", end='')
#     print("")
# # # # # ###################################################
# # # # #
# # # # #
# # # # # ####################################################################
# # # # Create soccer game MG (inherits from MDP class)
# print("Begin Soccer VI")
# print("THIS CURRENT SETTING WORKS")
# soccer = SoccerGame()

# Create LONR agent, feed in soccer
# A25
# ** | ** | ** | ** | ** | **
# ** | gB |    | A+ | gA | **
# ** | gB | B  |    | gA | **
# ** | ** | ** | ** | ** | **

# FIX TODAY
# lonrAgent = LONR(M=soccer, alpha=0.99, gamma=0.99, alphaDecay=1.0, RMPLUS=False, DCFR=True, VI=True)
#
# # Train via VI (4500 to get PI to converge)
# lonrAgent.lonr_value_iteration(iterations=4500, log=50)
#
# # Test of the learned policy:
#
# # Normalize pi sums here for now
# # soccer.normalize_pisums()
# # print("A21pisums Player A", soccer.pi_sums[0]["A21"])
# # print("A21Q  Player A", soccer.Q[0]["A21"])
# # print("")
# # print("A21pisums Player B", soccer.pi_sums[1]["A21"])
# # print("A21Q  Player B", soccer.Q[1]["A21"])
#
# print("")
# print("A25pisums Player A", soccer.pi_sums[0]["A25"])
# print("A25Q  Player A", soccer.Q[0]["A25"])
# print("")
# print("A25pisums Player B", soccer.pi_sums[1]["A25"])
# print("A25Q  Player B", soccer.Q[1]["A25"])


#
# # This plays face-to-face start state, player A has the ball
# # This is 66/33 win/lose for playerA. Should double check that is correct
# #
# print("Playing 50,000 games where:")
# print(" - Players are facing each other")
# print(" - Player A always starts with ball")
# soccer.play(iterations=50000, log=10000)
#
# print("")
#
# # Play random games, mix who has ball at start, positions, etc
# print("Playing 50,000 games where: all initial conditions are randomized")
# soccer.play_random(iterations=50000,log=10000)
#
# print("")
#
# print("Print out of game states of interest, here is board for reminder")
#
# print(" 0  1  2  3")
# print(" 4  5  6  7")
# print("")
# print("Note: [North, South, East, West]")
# print("B has ball in 6, A in 2")
# print("PiB26 A: ", soccer.pi[0]["B26"])
# print("PiB26 B: ", soccer.pi[1]["B26"])
# print("")
# print("B has ball in 5, A in 2")
# print("PiB25 A: ", soccer.pi[0]["B25"])
# print("PiB25 B: ", soccer.pi[1]["B25"])
#
# print("")
# print("A has ball in 2, B in 5")
# print("PiA25 A: ", soccer.pi[0]["A25"])
# print("PiA25 B: ", soccer.pi[1]["A25"])
# print("")
# print("A has ball in 1, B in 5")
# print("PiA15 A: ", soccer.pi[0]["A15"])
# print("PiA15 B: ", soccer.pi[1]["A15"])
# print("")
# print("End Soccer VI")
# print("")
# # ######################################################################
# #
# #
# # ######################################################################
# # # NoSDE
# # #
# # # SET GAMMA = 0.75 (THIS IS REQUIRED)
# # # ALPHA works with 0.5, iterations 400k +
# # # Better results when alpha is decayed
# # #
# #
# print("Begin NoSDE VI")
# noSDE = NoSDE()
#
# # Create LONR agent, feed in soccer (gamma should be 0.75 here)
# # DCFR had to be added here, as it is the only one that converges!
# lonrAgent = LONR(M=noSDE, gamma=0.75, alpha=0.5, alphaDecay=0.9999, DCFR=True, VI=True)
#
# print(" - Training with 50000 iterations")
# lonrAgent.lonr_value_iteration(iterations=20000, log=5000)
#
#
# print("")
# print("Pi Sums:")
# for n in range(2):
#     for k in sorted(noSDE.pi_sums[n].keys()):
#        print(k, ": ", noSDE.pi_sums[n][k])
# print("")
# print("Normalized Pi Sums:")
# print(" - Player 1 Send with: 2/3 (0.666)")
# print(" - Player 2 Send with 5/12 (0.416")
# for n in range(2):
#     for k in sorted(noSDE.pi_sums[n].keys()):
#         #print(k, ": ", noSDE.pi_sums[k])
#         tot = 0.0
#         for kk in noSDE.pi_sums[n][k].keys():
#             tot += noSDE.pi_sums[n][k][kk]
#         print(k, ": ", end='')
#         for kk in noSDE.pi_sums[n][k].keys():
#             print(kk, ": ", noSDE.pi_sums[n][k][kk] / float(tot), " ", end='')
#         print("")
# print("")
# print("QValues:")
# print("  - Should be 4 every SEND/KEEP")
# print("  - Should be 5.3 every NOOP")
# for n in range(2):
#     for k in sorted(noSDE.Q[n].keys()):
#         #print(k, ": ", noSDE.pi_sums[k])
#         print(k, ": ", end='')
#         for kk in noSDE.Q[n][k].keys():
#             print(kk, ": ", noSDE.Q[n][k][kk], " ", end='')
#         print("")
# print("")
# print("End NoSDE VI")


######################################################################
# Tiger Game AVI
#####################################################################
# print("")
# tigerGame = TigerGame(startState="START", TLProb=0.5, version=1)
#
#
# # # # Create LONR Agent and feed in the Tiger game
# parameters = {'alpha': 1.0, 'epsilon': 10, 'gamma': 1.0}
# regret_minimizers = {'RM': True, 'RMPlus': False, 'DCFR': False}
# # #
# # #
# lonrAgent = LONR_A(M=tigerGame, parameters=parameters, regret_minimizers=regret_minimizers)
#
# iters = 100000
# lonrAgent.lonr_train(iterations=iters, log=2500, randomize=True)
#
# print("DONT FORGET START STATE")
# #
#
#
# print("")
# print("Pi sums: ")
# for k in sorted(tigerGame.pi_sums[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.pi_sums[0][k].keys()):
#         print(kk, ": ", tigerGame.pi_sums[0][k][kk], " ", end="")
#     print("")
# #
# print("")
# print("Q: ")
# for k in sorted(tigerGame.Q[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.Q[0][k].keys()):
#         print(kk, ": ", tigerGame.Q[0][k][kk], " ", end="")
#     print("")
#
#
# print("")
# print("Pi : ")
# for k in sorted(tigerGame.pi[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.pi[0][k].keys()):
#         print(kk, ": ", tigerGame.pi[0][k][kk], " ", end="")
#     print("")
#
# #
# print("")
# print("Q Avg")
# for k in sorted(tigerGame.QSums[0].keys()):
#     tot = 0.0
#     # for kk in agent.Qsums[k].keys():
#     #     tot += agent.Qsums[k][kk]
#     # if tot == 0:
#     #     tot = 1.0
#     print(k, ": ", end='')
#     for kk in tigerGame.QSums[0][k].keys():
#         print(kk, ": ", tigerGame.QSums[0][k][kk] / tigerGame.QTouched[0][k][kk], " ", end='')
#     print("")
#
#
# print("")
# print("Regret sums : ")
# for k in sorted(tigerGame.regret_sums[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.regret_sums[0][k].keys()):
#         print(kk, ": ", tigerGame.regret_sums[0][k][kk], " ", end="")
#     print("")


# print("----------------------------------------------")
# print("    End of Value Iteration Tests")
# print("----------------------------------------------")
# print("")
# # #######################################################################
#
#
# print("----------------------------------------------")
# print("   Begin O-LONR Tests")
# print("----------------------------------------------")
# print("")

####################################################################




#################################################################################
# START OF EXP 3
# change grid, rows, cols, living cost in GridWorld.py

# O-LONR GridWorld
# print("Begin GridWorld O-LONR - deterministic")
# print("THIS WORKS MUCH BETTER WITH RM+ - undo past large negative regret sums")
# print("THIS grid will work with vanilla RM, just takes 30k iterations or so")
# gridMDP = Grid(noise=0.0, startState=6)
#
# # Create LONR Agent and feed in gridMDP
# ###QLEARNlonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=40, alphaDecay=1.0, RMPLUS=False, DCFR=False, VI=False, LONR=False)
# lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=20, alphaDecay=1.0, RMPLUS=False, DCFR=False, VI=False, EXP3=True, exp3gamma=0.15)
#
# # Train via AVI
# totalIters = 1
# lonrAgent.lonr_online(iterations=totalIters, log=500)
#
# # lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=20, alphaDecay=1.0, RMPLUS=False, DCFR=False, VI=False, EXP3=True, exp3gamma=0.0)
# #
# # # Train via VI
# # totalIters = 23400
# # lonrAgent.lonr_online(iterations=totalIters, log=1200)
#
# print("Pi")
# for k in gridMDP.pi[0].keys():
#     print("Pi: ", k, ": ", gridMDP.pi[0][k])
#
# for k in gridMDP.Q[0].keys():
#     print("Q: ", k, ": ", gridMDP.Q[0][k])
#
# print("")
# print("Q values (not QAvg)")
# print("Bottom left (start state)")
# print(gridMDP.Q[0][6])
#
# print("State above bottom right terminal")
# print(gridMDP.Q[0][6])
#
# print("PI Sums")
# for k in gridMDP.pi_sums[0].keys():
#     print("PISUM: ", k, " ( ", end='')
#     total = 0.0
#     for kk in gridMDP.pi_sums[0][k].keys():
#         total += gridMDP.pi_sums[0][k][kk]
#     if total <= 0: total = 1
#     for kk in gridMDP.pi_sums[0][k].keys():
#         print(kk, ": ", gridMDP.pi_sums[0][k][kk] / total, " ", end='')
#     print(" )")
#
#
# print("")
# print("End GridWorld O-LONR - deterministic")



## END OF EXP3
#######################################################################################

#
# print("")
#
# print("Begin GridWorld O-LONR - non-deterministic")
# gridMDP = Grid(noise=0.2, startState=36)
#
# # Create LONR Agent and feed in gridMDP
# lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=15, alphaDecay=0.999, VI=False)
#
# # Train via VI
# lonrAgent.lonr_online(iterations=20000, log=500, randomized=False)
#
# print("")
# print("Q values (not QAvg)")
# print("Bottom left (start state)")
# print(gridMDP.Q[0][36])
#
# print("State above bottom right terminal")
# print(gridMDP.Q[0][35])
#
# print("End GridWorld O-LONR - non-deterministic")
# print("")
#####################################################################


# Tiger Game O-LONR - Tiger Location 50/50
# print("Begin TigerGame O-LONR - Tiger Location 50/50")
# tigerGame = TigerGame(startState="root", TLProb=0.5)
#
# # # Create LONR Agent and feed in the Tiger game
# lonrAgent = LONR(M=tigerGame, gamma=1.0, alpha=0.5, epsilon=20, alphaDecay=1.0, RMPLUS=False, VI=False, randomize=True)
#
# lonrAgent.lonr_online(iterations=20111, log=1000, randomized=True)
#
# print("")
# print("Pi sums: ")
# for k in sorted(tigerGame.pi_sums[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.pi_sums[0][k].keys()):
#         print(kk, ": ", tigerGame.pi_sums[0][k][kk], " ", end="")
#     print("")
#
# print("")
# print("Q: ")
# for k in sorted(tigerGame.Q[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.Q[0][k].keys()):
#         print(kk, ": ", tigerGame.Q[0][k][kk], " ", end="")
#     print("")
#
# print("")

# Tiger Game O-LONR - Tiger Location 85/15
# print("Begin TigerGame O-LONR - Tiger Location 85/15")
# tigerGame = TigerGame(startState="root", TLProb=0.85)
#
# # Create LONR Agent and feed in the Tiger game
# lonrAgent = LONR(M=tigerGame, gamma=1.0, alpha=0.5, epsilon=15, alphaDecay=0.999)
#
# lonrAgent.lonr_online(iterations=20000, log=2000, randomized=True)
#
# print("")
# print("Pi sums: ")
# for k in sorted(tigerGame.pi_sums[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.pi_sums[0][k].keys()):
#         print(kk, ": ", tigerGame.pi_sums[0][k][kk], " ", end="")
#     print("")
#
# print("")
# print("Q: ")
# for k in sorted(tigerGame.Q[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.Q[0][k].keys()):
#         print(kk, ": ", tigerGame.Q[0][k][kk], " ", end="")
#     print("")





######################################################
# Exp3
# O-LONR GridWorld
# print("Begin GridWorld O-LONR - deterministic")
# print("THIS WORKS MUCH BETTER WITH RM+ - undo past large negative regret sums")
# print("THIS grid will work with vanilla RM, just takes 30k iterations or so")
# gridMDP = Grid(noise=0.0, startState=12)
#
# Make Sure to Set EXP bool
# # Create LONR Agent and feed in gridMDP
# lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=20, alphaDecay=1.0, RMPLUS=False, DCFR=False, VI=False)
#
# # Train via VI
# totalIters = 40001
# lonrAgent.lonr_online(iterations=totalIters, log=2000)
#
# # for k in gridMDP.pi[0].keys():
# #     print("PI: ", k, ": ", gridMDP.pi[0][k])
# #
# # for k in gridMDP.regret_sums[0].keys():
# #     print("regretSums: ", k, ": ", gridMDP.regret_sums[0][k])
#
#
# #print("Q values (not QAvg)")
# print("QAvg for Bottom left (start state)")
# for a in sorted(gridMDP.QSums[0][12].keys()):
#     print(a, ": ", gridMDP.QSums[0][12][a] / float(totalIters))
#
# print("Q for Bottom left (start state)")
# for a in sorted(gridMDP.Q[0][12].keys()):
#     print(gridMDP.Q[0][12][a])
#
# print("Pi for Bottom left (start state)")
# for a in sorted(gridMDP.pi[0][12].keys()):
#     print(gridMDP.pi[0][12][a])
# print("")
# print("Pi:")
# for s in gridMDP.pi[0].keys():
#     print(s, ": ", gridMDP.pi[0][s])
# print("")
# print("Pi sums:")
# for s in gridMDP.pi_sums[0].keys():
#     print(s, ": ", gridMDP.pi_sums[0][s])
#
# print("Weights:")
# for s in sorted(gridMDP.weights[0].keys()):
#     print(s, ": ", gridMDP.weights[0][s])
#
#
# print("Q for State above bottom right terminal")
# #print(gridMDP.QSums[0][35])
#
# print("End GridWorld O-LONR - deterministic")


print("Begin GridWorld VI with non-determinism")
gridMDP = Grid(noise=0.0, startState=4)

# Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
parameters = {'alpha': 1.0, 'epsilon': 20, 'gamma': 1.0}
regret_minimizers = {'RM': True, 'RMPlus': True, 'DCFR': False}
lonrAgent = LONR_B(M=gridMDP, parameters=parameters, regret_minimizers=regret_minimizers)

# Train via VI
iters = 33
lonrAgent.lonr_train(iterations=iters, log=500)

print("[North, East, South, West]")
print("Note: these are Q, not QAvg")
print("Q for: Bottom left (start state) (West opt = 170.23]")
print(gridMDP.Q[0][4])

print("Q for: State above bottom right terminal")
print(gridMDP.Q[0][3])
print("End GridWorld VI with non-determism")
print("")
print("Q Avg")
for k in sorted(gridMDP.QSums[0].keys()):
    tot = 0.0
    print(k, ": ", end='')
    for kk in gridMDP.QSums[0][k].keys():
        touched = gridMDP.QTouched[0][k][kk]
        if touched == 0: touched = 1.0
        print(kk, ": ", gridMDP.QSums[0][k][kk] / touched, " ", end='')
    print("")

print("")
print("Q")
for k in sorted(gridMDP.Q_bu[0].keys()):
    tot = 0.0
    print(k, ": ", end='')
    for kk in gridMDP.Q_bu[0][k].keys():
        print(kk, ": ", gridMDP.Q_bu[0][k][kk], " ", end='')
    print("")

print("")
print("Pi Sums")
for k in sorted(gridMDP.pi_sums[0].keys()):
    tot = 0.0
    for kk in gridMDP.pi_sums[0][k].keys():
        tot += gridMDP.pi_sums[0][k][kk]
    if tot == 0: tot = 1.0
    print(k, ": ", end='')
    for kk in gridMDP.pi_sums[0][k].keys():
        print(kk, ": ", gridMDP.pi_sums[0][k][kk] / tot, " ", end='')
    print("")

print("")
print("Pi")
for k in sorted(gridMDP.pi[0].keys()):
    tot = 0.0
    for kk in gridMDP.pi[0][k].keys():
        tot += gridMDP.pi[0][k][kk]
    if tot == 0: tot = 1.0
    print(k, ": ", end='')
    for kk in gridMDP.pi[0][k].keys():
        print(kk, ": ", gridMDP.pi[0][k][kk], " ", end='')
    print("")




print("Done")





















# for s in self.total_states.keys():
#
#     if self.isTerminal(s): continue
#
#     player_with_ball = s[0]
#
#     pAball = False
#     pBball = False
#     if player_with_ball == "A":
#         pAball = True
#     if player_with_ball == "B":
#         pBball = True
#
#     playerApos = int(s[1])
#     playerBpos = int(s[2])
#
#     pAx, pAy = stateToXY(playerApos)
#     pBx, pBy = stateToXY(playerBpos)
#
#     for actions_A in self.total_actions:
#         QV = 0.0
#         oppValue = 0.0
#         for actions_B in self.total_actions:
#
#             self.counter += 1
#
#             player_a = Player(x=pAx, y=pAy, has_ball=pAball, p_id='A')
#             player_b = Player(x=pBx, y=pBy, has_ball=pBball, p_id='B')
#
#             world = World()
#             world.set_world_size(x=self.cols, y=self.rows)
#             world.place_player(player_a, player_id='A')
#             world.place_player(player_b, player_id='B')
#             world.set_goals(100, 0, 'A')
#             world.set_goals(100, 3, 'B')
#
#             actions = {'A': actions_A, 'B': actions_B}
#             new_state, rewards, goal = world.move(actions)
#
#             rA = rewards['A']
#
#             Value = 0.0
#
#             for action_ in self.total_actions:
#                 Value += self.QvaluesA[new_state][action_] * self.piA[new_state][action_]
#             QV += self.piB[s][actions_B] * (rA + self.gamma * Value)
#
#         self.QValue_backupA[s][actions_A] = (0.1 * self.QvaluesA[s][actions_A]) + (0.9 * QV)
#
#
# #####################################################################
# #MDP
#     for s in range(self.numberOfStates):
#
#             # Terminal - Set Q as Reward (no actions in terminal states)
#             # - alternatively, one action which is 'exit' which gets full reward
#             if type(self.gridWorld.grid[s]) != str:
#                 for a in range(self.numberOfActions):
#                     self.Q_bu[s][a] = self.gridWorld.getReward(s)
#                 continue
#
#             # For walls , possibly (to conform to other setups)
#             # if grid[s] == '#':
#             #     continue
#
#             # Loop through all actions in current state s
#             for a in range(self.numberOfActions):
#
#
#                 succs = self.gridWorld.getNextStatesAndProbs(s,a)
#
#
#                 Value = 0.0
#                 for _, s_prime, prob in succs:
#                     tempValue = 0.0
#                     for a_prime in range(self.numberOfActions):
#                         tempValue += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
#                     Value += prob * (self.gridWorld.getReward(s) + self.gamma * tempValue)
#
#                 self.Q_bu[s][a] = Value
#
#
# ###############################################################
# # General LONR
#
#     for n in range(N): #for every players
#
#         for s in range(S): #For every state
#
#             for action_current in range(A(n)): # for actions in player n
#
#                 # Get successor states
#                 # For 1 player mdp, other player = determism/non-determinism
#                 succs = getProbsAndNextStates(s, p_n, p_not_n)
#
#                 Value = 0.0
#                 # For non-determ mdp, prob is stochasticity
#                 # For 2 player MG, prob is other agent
#                 for _, s_prime, prob in succs:
#                     tempValue = 0.0
#                     for a_prime in range(self.numberOfActions):
#                         tempValue += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
#                     Value += prob * (self.gridWorld.getReward(s) + self.gamma * tempValue)
#
#                 self.QValue_backupA[s][actions_A] = (0.1 * self.QvaluesA[s][actions_A]) + (0.9 * QV)