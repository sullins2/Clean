from LONRGeneral.LONR import *
from LONRGeneral.GridWorld import *
from LONRGeneral.Soccer import *
from LONRGeneral.NoSDE import *

####################################################
# GridWorld MDP
#########
# create GridWorld (inherits from MDP class)
# gridMDP = Grid(noise=0.2)
#
# # Create LONR Agent and feed in gridMDP
# lonrAgent = LONR(M=gridMDP, gamma=1.0)
#
# # Train via VI
# lonrAgent.lonr_value_iteration(iterations=1000, log=100)
#
# print("Bottom left (start state)")
# print(gridMDP.Q[0][36])
#
# print("State above bottom right terminal")
# print(gridMDP.Q[0][35])
# print("")
###################################################


####################################################################
# Create soccer game MG (inherits from MDP class)
# soccer = SoccerGame()
#
# # Create LONR agent, feed in soccer
# lonrAgent = LONR(M=soccer, gamma=0.95)
#
# # Train via VI
# lonrAgent.lonr_value_iteration(iterations=1, log=500)
#
# # Test of the learned policy:
#
# # Normalize pi sums here for now
# soccer.normalize_pisums()
#
# # This plays face-to-face start state, player A has the ball
# # This is 66/33 win/lose for playerA. Should double check that is correct
# #
# #soccer.play(iterations=1, log=10000)
#
# print("")
#
# # Play random games, mix who has ball at start, positions, etc
# #soccer.play_random(iterations=1,log=10000)
#
# print("")
#
# print("Print out of game states of interest")
#
# print(" 0  1  2  3")
# print(" 4  5  6  7")
# print("")
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

######################################################################

# SET GAMMA = 0.75

noSDE = NoSDE()

# Create LONR agent, feed in soccer
lonrAgent = LONR(M=noSDE, gamma=0.75, alpha=0.99)


lonrAgent.lonr_value_iteration(iterations=450000, log=1000)



print("")
print("Normalized Pi Sums:")
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
for n in range(2):
    for k in sorted(noSDE.Q[n].keys()):
        #print(k, ": ", noSDE.pi_sums[k])
        print(k, ": ", end='')
        for kk in noSDE.Q[n][k].keys():
            print(kk, ": ", noSDE.Q[n][k][kk], " ", end='')
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