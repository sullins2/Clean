from LONRGeneral.LONR import *
from LONRGeneral.GridWorld import *
from LONRGeneral.Soccer import *
from LONRGeneral.NoSDE import *
from LONRGeneral.TigerGame import *

####################################################
# GridWorld MDP
####################################################

# Create GridWorld (inherits from MDP class)
print("Begin GridWorld VI with determinism")
gridMDP = Grid(noise=0.0)

# Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, DCFR=False, VI=True)

# Train via VI
lonrAgent.lonr_value_iteration(iterations=150, log=10)
print("[North, East, South, West]")
print("Note: these are Q, not QAvg")
print("Q for: Bottom left (start state)")
print(gridMDP.Q[0][36])

print("Q for: State above bottom right terminal")
print(gridMDP.Q[0][35])
print("End GridWorld VI with determism")
print("")


# # Create GridWorld (inherits from MDP class)
print("Begin GridWorld VI with non-determinism")
gridMDP = Grid(noise=0.2)

# Create LONR Agent and feed in gridMDP (alpha should be 1.0 here)
lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, VI=True)

# Train via VI
lonrAgent.lonr_value_iteration(iterations=1000, log=1000)

print("[North, East, South, West]")
print("Note: these are Q, not QAvg")
print("Q for: Bottom left (start state) (West opt = 170.23]")
print(gridMDP.Q[0][36])

print("Q for: State above bottom right terminal")
print(gridMDP.Q[0][35])
print("End GridWorld VI with non-determism")
print("")
# # # # ###################################################
# # # #
# # # #
# # # # ####################################################################
# # # Create soccer game MG (inherits from MDP class)
print("Begin Soccer VI")
print("THIS CURRENT SETTING WORKS")
soccer = SoccerGame()

# Create LONR agent, feed in soccer
lonrAgent = LONR(M=soccer, alpha=1.0, gamma=0.95, RMPLUS=False, DCFR=False, VI=True)

# Train via VI
lonrAgent.lonr_value_iteration(iterations=4500, log=50)

# Test of the learned policy:

# Normalize pi sums here for now
soccer.normalize_pisums()

# This plays face-to-face start state, player A has the ball
# This is 66/33 win/lose for playerA. Should double check that is correct
#
print("Playing 50,000 games where:")
print(" - Players are facing each other")
print(" - Player A always starts with ball")
soccer.play(iterations=50000, log=10000)

print("")

# Play random games, mix who has ball at start, positions, etc
print("Playing 50,000 games where: all initial conditions are randomized")
soccer.play_random(iterations=50000,log=10000)

print("")

print("Print out of game states of interest, here is board for reminder")

print(" 0  1  2  3")
print(" 4  5  6  7")
print("")
print("Note: [North, South, East, West]")
print("B has ball in 6, A in 2")
print("PiB26 A: ", soccer.pi[0]["B26"])
print("PiB26 B: ", soccer.pi[1]["B26"])
print("")
print("B has ball in 5, A in 2")
print("PiB25 A: ", soccer.pi[0]["B25"])
print("PiB25 B: ", soccer.pi[1]["B25"])

print("")
print("A has ball in 2, B in 5")
print("PiA25 A: ", soccer.pi[0]["A25"])
print("PiA25 B: ", soccer.pi[1]["A25"])
print("")
print("A has ball in 1, B in 5")
print("PiA15 A: ", soccer.pi[0]["A15"])
print("PiA15 B: ", soccer.pi[1]["A15"])
print("")
print("End Soccer VI")
print("")
# ######################################################################
#
#
# ######################################################################
# # NoSDE
# #
# # SET GAMMA = 0.75 (THIS IS REQUIRED)
# # ALPHA works with 0.5, iterations 400k +
# # Better results when alpha is decayed
# #
#
print("Begin NoSDE VI")
noSDE = NoSDE()

# Create LONR agent, feed in soccer (gamma should be 0.75 here)
# DCFR had to be added here, as it is the only one that converges!
lonrAgent = LONR(M=noSDE, gamma=0.75, alpha=0.5, alphaDecay=0.9999, DCFR=True, VI=True)

print(" - Training with 50000 iterations")
lonrAgent.lonr_value_iteration(iterations=20000, log=5000)


print("")
print("Pi Sums:")
for n in range(2):
    for k in sorted(noSDE.pi_sums[n].keys()):
       print(k, ": ", noSDE.pi_sums[n][k])
print("")
print("Normalized Pi Sums:")
print(" - Player 1 Send with: 2/3 (0.666)")
print(" - Player 2 Send with 5/12 (0.416")
for n in range(2):
    for k in sorted(noSDE.pi_sums[n].keys()):
        #print(k, ": ", noSDE.pi_sums[k])
        tot = 0.0
        for kk in noSDE.pi_sums[n][k].keys():
            tot += noSDE.pi_sums[n][k][kk]
        print(k, ": ", end='')
        for kk in noSDE.pi_sums[n][k].keys():
            print(kk, ": ", noSDE.pi_sums[n][k][kk] / float(tot), " ", end='')
        print("")
print("")
print("QValues:")
print("  - Should be 4 every SEND/KEEP")
print("  - Should be 5.3 every NOOP")
for n in range(2):
    for k in sorted(noSDE.Q[n].keys()):
        #print(k, ": ", noSDE.pi_sums[k])
        print(k, ": ", end='')
        for kk in noSDE.Q[n][k].keys():
            print(kk, ": ", noSDE.Q[n][k][kk], " ", end='')
        print("")
print("")
print("End NoSDE VI")


######################################################################
# Tiger Game VI
#####################################################################
# tigerGame = TigerGame(startState="root", TLProb=0.5)
#
# # # Create LONR Agent and feed in the Tiger game
# lonrAgent = LONR(M=tigerGame, gamma=1.0, alpha=1.0, epsilon=15, alphaDecay=0.999, RMPLUS=False, DCFR=True, VI=True, randomize=True)
#
# lonrAgent.lonr_value_iteration(iterations=2222, log=1000)
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
#
# print("")
# print("Pi : ")
# for k in sorted(tigerGame.pi[0].keys()):
#     print(k, ": ", end='')
#     for kk in sorted(tigerGame.pi[0][k].keys()):
#         print(kk, ": ", tigerGame.pi[0][k][kk], " ", end="")
#     print("")

#
print("----------------------------------------------")
print("    End of Value Iteration Tests")
print("----------------------------------------------")
print("")
# #######################################################################


print("----------------------------------------------")
print("   Begin O-LONR Tests")
print("----------------------------------------------")
print("")

####################################################################
# O-LONR GridWorld
print("Begin GridWorld O-LONR - deterministic")
print("THIS WORKS MUCH BETTER WITH RM+ - undo past large negative regret sums")
print("THIS grid will work with vanilla RM, just takes 30k iterations or so")
gridMDP = Grid(noise=0.0, startState=36)

# Create LONR Agent and feed in gridMDP
lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=20, alphaDecay=1.0, RMPLUS=True, DCFR=False, VI=False)

# Train via VI
lonrAgent.lonr_online(iterations=10000, log=1000)

for k in gridMDP.pi[0].keys():
    print("PI: ", k, ": ", gridMDP.pi[0][k])

for k in gridMDP.regret_sums[0].keys():
    print("regretSums: ", k, ": ", gridMDP.regret_sums[0][k])


print("Q values (not QAvg)")
print("Q for Bottom left (start state)")
print(gridMDP.Q[0][36])

print("Q for State above bottom right terminal")
print(gridMDP.Q[0][35])

print("End GridWorld O-LONR - deterministic")

print("")

print("Begin GridWorld O-LONR - non-deterministic")
gridMDP = Grid(noise=0.2, startState=36)

# Create LONR Agent and feed in gridMDP
lonrAgent = LONR(M=gridMDP, gamma=1.0, alpha=1.0, epsilon=15, alphaDecay=0.999, VI=False)

# Train via VI
lonrAgent.lonr_online(iterations=20000, log=500, randomized=False)

print("")
print("Q values (not QAvg)")
print("Bottom left (start state)")
print(gridMDP.Q[0][36])

print("State above bottom right terminal")
print(gridMDP.Q[0][35])

print("End GridWorld O-LONR - non-deterministic")
print("")
#####################################################################


# Tiger Game O-LONR - Tiger Location 50/50
print("Begin TigerGame O-LONR - Tiger Location 50/50")
tigerGame = TigerGame(startState="root", TLProb=0.5)

# # Create LONR Agent and feed in the Tiger game
lonrAgent = LONR(M=tigerGame, gamma=1.0, alpha=0.5, epsilon=20, alphaDecay=1.0, RMPLUS=False, VI=False, randomize=True)

lonrAgent.lonr_online(iterations=15000, log=1000, randomized=True)

print("")
print("Pi sums: ")
for k in sorted(tigerGame.pi_sums[0].keys()):
    print(k, ": ", end='')
    for kk in sorted(tigerGame.pi_sums[0][k].keys()):
        print(kk, ": ", tigerGame.pi_sums[0][k][kk], " ", end="")
    print("")

print("")
print("Q: ")
for k in sorted(tigerGame.Q[0].keys()):
    print(k, ": ", end='')
    for kk in sorted(tigerGame.Q[0][k].keys()):
        print(kk, ": ", tigerGame.Q[0][k][kk], " ", end="")
    print("")

print("")
#
# Tiger Game O-LONR - Tiger Location 85/15
print("Begin TigerGame O-LONR - Tiger Location 85/15")
tigerGame = TigerGame(startState="root", TLProb=0.85)

# # Create LONR Agent and feed in the Tiger game
lonrAgent = LONR(M=tigerGame, gamma=1.0, alpha=1.0, epsilon=15, alphaDecay=0.999)

lonrAgent.lonr_online(iterations=20000, log=2000, randomized=True)

print("")
print("Pi sums: ")
for k in sorted(tigerGame.pi_sums[0].keys()):
    print(k, ": ", end='')
    for kk in sorted(tigerGame.pi_sums[0][k].keys()):
        print(kk, ": ", tigerGame.pi_sums[0][k][kk], " ", end="")
    print("")

print("")
print("Q: ")
for k in sorted(tigerGame.Q[0].keys()):
    print(k, ": ", end='')
    for kk in sorted(tigerGame.Q[0][k].keys()):
        print(kk, ": ", tigerGame.Q[0][k][kk], " ", end="")
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