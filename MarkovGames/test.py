from MarkovGames.soccer import *
import numpy as np
import copy


# def print_status(goal, new_state, rewards, total_states):
#     print("")
#     print("Players state label: {}".format(new_state))
#     print("Players state in the numerical table: {}".format(total_states[new_state]))
#     print("Rewards for each player after move: {}".format(rewards))
#     print("Goal status: {}".format(goal))
#     print("-" * 20 + "\n")


def main():



    lonrAgent = Agent()

    lonrAgent.train_lonr(iterations=1500, log=50)

    lonrAgent.test_lonr(iterations=50000, log=5000)

    lonrAgent.test_lonr_random_games(iterations=50000, log=5000)


    print("Print out of game states of interest")
    print("PiB26 A: ", lonrAgent.pi_sumsA["B26"])
    print("PiB26 B: ", lonrAgent.pi_sumsB["B26"])
    print("")
    print("PiB25 A: ", lonrAgent.pi_sumsA["B25"])
    print("PiB25 B: ", lonrAgent.pi_sumsB["B25"])
    print("")
    print("Counter: ", lonrAgent.counter)
    # print(lonrAgent.pi_sumsA)
    # print(lonrAgent.pi_sumsB)


    # print("QB: ", QvaluesB["B21"])
    # print("PB: ", pi_sumsB["B21"])
    # print("")
    # print("QA: ", QvaluesA["B21"])
    # print("PA: ", pi_sumsA["B21"])
    #
    # print("")
    # print("Random games: ")
    # playerARewards = 0.0
    # playerBRewards = 0.0
    # playerAWins = 0
    # playerBWins = 0
    # RANDOM_GAMES = 500
    # GameMoves = []
    # for iters in range(RANDOM_GAMES):
    #     print("")
    #     print("Game Number: ", iters)
    #     if iters % 500 == 0:
    #         print("Random game iteration: ", iters)
    #
    #
    #     validPos = [1, 2, 5, 6]
    #     ballStart = np.random.uniform(0, 1)
    #     AhasBall = True if ballStart < 0.5 else False
    #     BhasBall = False if ballStart < 0.5 else True
    #     playerAStart = validPos.pop(np.random.randint(0, 4))
    #     playerBStart = validPos.pop(np.random.randint(0, 3))
    #
    #     pAx, pAy = stateToXY(playerAStart)
    #     pBx, pBy = stateToXY(playerBStart)
    #
    #     player_a = Player(x=pAx, y=pAy, has_ball=AhasBall, p_id='A')
    #     player_b = Player(x=pBx, y=pBy, has_ball=BhasBall, p_id='B')
    #
    #     world = World()
    #     world.set_world_size(x=cols, y=rows)
    #     world.place_player(player_a, player_id='A')
    #     world.place_player(player_b, player_id='B')
    #     world.set_goals(100, 0, 'A')
    #     world.set_goals(100, 3, 'B')
    #
    #     curState = world.map_player_state()
    #
    #     world.plot_grid()
    #
    #     gms = 0
    #
    #     goal = False
    #     while not goal:
    #
    #         # 0 1 2 3
    #         # 4 5 6 7
    #
    #         gms += 1
    #         piAA = [pi_sumsA[curState][0], pi_sumsA[curState][1], pi_sumsA[curState][2], pi_sumsA[curState][3]]  # ,pi_sumsA[curState][4]]
    #         piBB = [pi_sumsB[curState][0], pi_sumsB[curState][1], pi_sumsB[curState][2], pi_sumsB[curState][3]]  # ,pi_sumsB[curState][4]]
    #         # pAa = np.random.choice([0,1,2,3,4], p=piAA)
    #         # pBa = np.random.choice([0,1,2,3,4], p=piBB)
    #         pAa = np.random.choice([0, 1, 2, 3], p=piAA)
    #         pBa = np.random.choice([0, 1, 2, 3], p=piBB)
    #         actions = {'A': pAa, 'B': pBa}
    #         new_state, rewards, goal = world.move(actions)
    #         playerARewards += rewards["A"]
    #         playerBRewards += rewards["B"]
    #         curState = new_state
    #         if int(rewards["A"]) == 100:
    #             playerAWins += 1
    #         if int(rewards["B"]) == 100:
    #             playerBWins += 1
    #         #world.plot_grid()
    #         #print_status(goal, new_state, rewards, total_states)
    #
    #     GameMoves.append(gms)
    # print("Final A: ", playerARewards)
    # print("Final B: ", playerBRewards)
    # print("Player A WINS: ", playerAWins)
    # print("Player B WINS: ", playerBWins)
    #
    # # 0 1 2 3
    # # 4 5 6 7
    # # B26
    # print("PiB26 A: ", pi_sumsA["B26"])
    # print("PiB26 B: ", pi_sumsB["B26"])
    # print("")
    # print("PiB25 A: ", pi_sumsA["B25"])
    # print("PiB25 B: ", pi_sumsB["B25"])
    # print("")
    # print(pi_sumsA)
    # print(pi_sumsB)
    # print("")
    # print("Game moves: ")
    # print(GameMoves)

    print("")
    print("Done")







































































# for s in total_states.keys(): #A01, etc
    #     QvaluesA[s] = {}
    #     QvaluesB[s] = {}
    #     QValue_backupA[s] = {}
    #     QValue_backupB[s] = {}
    #     piA[s] = {}
    #     piB[s] = {}
    #     regret_sumsA[s] = {}
    #     regret_sumsB[s] = {}
    #     pi_sumsA[s] = {}
    #     pi_sumsB[s] = {}
    #     for a in total_actions:
    #         QvaluesA[s][a] = 0.0
    #         QvaluesB[s][a] = 0.0
    #         QValue_backupA[s][a] = 0.0
    #         QValue_backupB[s][a] = 0.0
    #         piA[s][a] = 1.0 / 4.0
    #         piB[s][a] = 1.0 / 4.0
    #         regret_sumsA[s][a] = 0.0
    #         regret_sumsB[s][a] = 0.0
    #         pi_sumsA[s][a] = 0.0
    #         pi_sumsB[s][a] = 0.0
    #
    # alpha = 0.9999
    # gamma = 0.79999
    #
    # verbose = False
    #
    # counter = 0
    # world = World()
    # world.set_world_size(x=cols, y=rows)
    # LEPS = 10
    # alt = 2
    # REW_GAMMA = 0.99
    #
    # # for iters in range(LEPS):
    # #
    # #     #LIVING_COST += 0.01
    # #     if iters % 100 == 0:
    # #         print("Iteration: ", iters)
    # #     ###############################################################
    # #     # Loop thru player A
    # #     ###############################################################
    # #
    # #     #alt = np.random.uniform(0, 1)
    # #     #if alt == 1:
    # #
    # #     for s in sorted(total_states.keys()):
    # #
    # #         #Skip B
    # #         # if s[0] == "B":
    # #         #     continue
    # #
    # #         player_with_ball = s[0]
    # #
    # #         pAball = False
    # #         pBball = False
    # #         if player_with_ball == "A":
    # #             pAball = True
    # #         if player_with_ball == "B":
    # #             pBball = True
    # #
    # #
    # #         playerApos = int(s[1])
    # #         playerBpos = int(s[2])
    # #
    # #         pAx, pAy = stateToXY(playerApos)
    # #         pBx, pBy = stateToXY(playerBpos)
    # #         QV = 0.0
    # #         for actions_A in total_actions:
    # #             QV = 0.0
    # #             oppValue = 0.0
    # #             for actions_B in total_actions:
    # #             #actions_2 = np.random.randint(0, 5)
    # #
    # #                 player_a = Player(x=pAx, y=pAy, has_ball=pAball, p_id='A')
    # #                 player_b = Player(x=pBx, y=pBy, has_ball=pBball, p_id='B')
    # #
    # #                 world = World()
    # #                 world.set_world_size(x=cols, y=rows)
    # #                 world.place_player(player_a, player_id='A')
    # #                 world.place_player(player_b, player_id='B')
    # #                 world.set_goals(100, 0, 'A')
    # #                 world.set_goals(100, 3, 'B')
    # #
    # #                 # ignore states where the start condintions are a terminal condition
    # #                 dumb_actions = {'A': 4, 'B': 4}
    # #                 new_state, rewards, goal = world.move(dumb_actions)
    # #                 if rewards["A"] != 0 or rewards["B"] != 0:
    # #                     #print("Start state is a terminal: ", s)
    # #                     #print("Continuing...")
    # #                     #print("")
    # #                     continue
    # #
    # #                 counter += 1
    # #                 #world.plot_grid()
    # #
    # #                 actions = {'A': actions_A, 'B': actions_B}
    # #                 new_state, rewards, goal = world.move(actions)
    # #
    # #                 if verbose:
    # #                     print("Evalutating: ", s)
    # #                     print("------------")
    # #                     gridP = "  0  1  2  3" + "\n" + "  4  5  6  7 "
    # #                     gridP = gridP.replace(str(playerApos), "A")
    # #                     gridP = gridP.replace(str(playerBpos), "B")
    # #                     # print("  0  1  2  3")
    # #                     # print("  4  5  6  7 ")
    # #                     print(gridP)
    # #                     print("------------")
    # #                     print("  PlayerA pos: ", playerApos, " Action: ", actString(actions_A))
    # #                     print("  PlayerB pos: ", playerBpos, " Action: ", actString(actions_B))
    # #                     print("  PlayerWBall: ", player_with_ball)
    # #                     print("  New State  : ", new_state)
    # #                     print("  Goal       : ", goal)
    # #                     print("  Rewards: ", rewards)
    # #                     #world.plot_grid()
    # #                     print("")
    # #
    # #                 rA = rewards['A']
    # #                 # if new_state[0] == "A":
    # #                 #     rA -= LIVING_COST
    # #                 Value = 0.0
    # #                 #Value = piA[s]
    # #                 for action_ in total_actions:
    # #                     Value += QvaluesA[new_state][action_] * piA[new_state][action_]
    # #                 #QValue_backupA[s][actions_A] += alpha * (piB[s][actions_B] * (rA + gamma * Value - QvaluesA[s][actions_A]))
    # #                 QV += piB[s][actions_B] * (rA + gamma * Value)
    # #                 #QV += piB[s][actions_B] * (rA + gamma * Value) #alpha * (piB[s][actions_B] * (gamma * Value))#(rA + gamma * Value))
    # #             QValue_backupA[s][actions_A] = (0.1 * QvaluesA[s][actions_A]) + (0.9 * QV) #QV #(0.25 * QvaluesA[s][actions_A]) + (0.75 * QV) #QV
    # #
    # #             # reward = 0.0
    # #             #
    # #             # Value = 0.0  # (3.0 * (7.0 / 12.0))# + (7.0 * (7.0 / 12.0))
    # #             # for action_ in range(2):
    # #             #     Value += Qvalues22[0][action_] * pi22[0][action_]
    # #             # QValue_backup21[0][a] = (pi11[0][KEEP] * (0.0 + gamma * Qvalues21[0][a])) + ((pi11[0][SEND]) * (3.0 + gamma * Value))
    # #
    # #             #QValue_backup12[0][a] = ((pi22[0][KEEP]) * (3.0 + gamma * Qvalues12[0][a])) + ((pi22[0][SEND]) *(gamma * Value))
    # #     #else:
    # #     ###############################################################
    # #     # Loop thru player B
    # #     ###############################################################
    # #     for s in sorted(total_states.keys()):
    # #
    # #         # Skip B
    # #         # if s[0] == "A":
    # #         #     continue
    # #
    # #         player_with_ball = s[0]
    # #
    # #         pAball = False
    # #         pBball = False
    # #         if player_with_ball == "A":
    # #             pAball = True
    # #         if player_with_ball == "B":
    # #             pBball = True
    # #
    # #         playerApos = int(s[1])
    # #         playerBpos = int(s[2])
    # #
    # #         pAx, pAy = stateToXY(playerApos)
    # #         pBx, pBy = stateToXY(playerBpos)
    # #         for actions_B in total_actions:
    # #
    # #             QV = 0.0
    # #             oppValue = 0.0
    # #             for actions_A in total_actions:
    # #                 # actions_2 = np.random.randint(0, 5)
    # #
    # #                 player_a = Player(x=pAx, y=pAy, has_ball=pAball, p_id='A')
    # #                 player_b = Player(x=pBx, y=pBy, has_ball=pBball, p_id='B')
    # #
    # #                 world = World()
    # #                 world.set_world_size(x=cols, y=rows)
    # #                 world.place_player(player_a, player_id='A')
    # #                 world.place_player(player_b, player_id='B')
    # #                 world.set_goals(100, 0, 'A')
    # #                 world.set_goals(100, 3, 'B')
    # #
    # #                 # ignore states where the start condintions are a terminal condition
    # #                 dumb_actions = {'A': 4, 'B': 4}
    # #                 new_state, rewards, goal = world.move(dumb_actions)
    # #                 if rewards["A"] != 0 or rewards["B"] != 0:
    # #                     # print("Start state is a terminal: ", s)
    # #                     # print("Continuing...")
    # #                     # print("")
    # #                     continue
    # #
    # #                 counter += 1
    # #                 # world.plot_grid()
    # #
    # #                 actions = {'A': actions_A, 'B': actions_B}
    # #                 new_state, rewards, goal = world.move(actions)
    # #
    # #                 if verbose:
    # #                     print("Evalutating: ", s)
    # #                     print("------------")
    # #                     gridP = "  0  1  2  3" + "\n" + "  4  5  6  7 "
    # #                     gridP = gridP.replace(str(playerApos), "A")
    # #                     gridP = gridP.replace(str(playerBpos), "B")
    # #                     # print("  0  1  2  3")
    # #                     # print("  4  5  6  7 ")
    # #                     print(gridP)
    # #                     print("------------")
    # #                     print("  PlayerA pos: ", playerApos, " Action: ", actString(actions_A))
    # #                     print("  PlayerB pos: ", playerBpos, " Action: ", actString(actions_B))
    # #                     print("  PlayerWBall: ", player_with_ball)
    # #                     print("  New State  : ", new_state)
    # #                     print("  Goal       : ", goal)
    # #                     print("  Rewards: ", rewards)
    # #                     # world.plot_grid()
    # #                     print("")
    # #
    # #                 rB = rewards['B']
    # #                 # if new_state[0] == "B":
    # #                 #     rB -= LIVING_COST
    # #                 Value = 0.0
    # #                 for action_ in total_actions:
    # #                     Value += QvaluesB[new_state][action_] * piB[new_state][action_]
    # #                 QV += piA[s][actions_A] * (rB + gamma * Value) #(piA[s][actions_A] * (gamma * Value))#(rB + gamma * Value))
    # #                 #QValue_backupB[s][actions_B] += alpha * (piA[s][actions_A] * (rB + gamma * Value - QvaluesB[s][actions_B]))
    # #                 #QV += alpha * (piA[s][actions_A] * (rB + gamma * Value - QvaluesB[s][actions_B]))
    # #             QValue_backupB[s][actions_B] = (0.1 * QvaluesB[s][actions_B]) + (0.9 * QV)#QV#(0.25 * QvaluesB[s][actions_B]) + (0.75 * QV) #(0.25 * QvaluesB[s][actions_B]) + (0.75 * QV) #QV
    # #                 # rA = rewards["A"]
    # #                 # Value = 0.0
    # #                 # for action_ in total_actions:
    # #                 #     Value += QvaluesA[new_state][action_] * piA[new_state][action_]
    # #                 # QValue_backupA[s][actions_A] += alpha * (piB[s][actions_B] * (rA + gamma * Value - QvaluesA[s][actions_A]))
    # #
    # #                 #QV += alpha * (piA[s][actions_B] * (rB + gamma * Value))
    # #             #QValue_backupB[s][actions_A] = QV
    # #                 # QValue_backup12[0][a] = ((pi22[0][KEEP]) * (3.0 + gamma * Qvalues12[0][a])) + ((pi22[0][SEND]) *(gamma * Value))
    # #
    # #     if alt == 1:
    # #         alt = 2
    # #     else:
    # #         alt = 1
    # #
    # #     QvaluesA = copy.deepcopy(QValue_backupA)
    # #     QvaluesB = copy.deepcopy(QValue_backupB)
    # #
    # #
    # #     #update regret / policy
    # #
    # #     # DCFR STUFF
    # #     alphaR = 3.0 / 2.0  # accum pos regrets
    # #     betaR = 0.0  # accum neg regrets
    # #     gammaR = 2.0  # contribution to avg strategy
    # #
    # #     alphaW = pow(iters, alphaR)
    # #     alphaWeight = (alphaW / (alphaW + 1))
    # #
    # #     betaW = pow(iters, betaR)
    # #     betaWeight = (betaW / (betaW + 1))
    # #
    # #     gammaWeight = pow((iters / (iters + 1)), gammaR)
    # #
    # #     # Update regret sums PLAYER 1
    # #     for s in sorted(QvaluesA.keys()):
    # #
    # #         target = 0.0
    # #         for a in total_actions:
    # #             target += QvaluesA[s][a] * piA[s][a]
    # #
    # #             # if target > 0.001:
    # #             #     print("Target: ", piA[s][a])
    # #         for a in total_actions:
    # #             action_regret = QvaluesA[s][a] - target  # max(0.0, Qvalues1[s][a] - target)
    # #             if action_regret > 0:
    # #                 action_regret *= alphaWeight
    # #             else:
    # #                 action_regret *= betaWeight
    # #
    # #             # if iters > 149000:
    # #             #     print("Regret on ", iters, ": ", action_regret)
    # #             regret_sumsA[s][a] += action_regret #max(0.0, action_regret) #action_regret
    # #
    # #     for s in sorted(piA.keys()):
    # #
    # #         rgrt_sum = 0.0
    # #         for a in total_actions:
    # #             rgrt_sum += max(0.0, regret_sumsA[s][a])
    # #
    # #         for a in total_actions:
    # #             #rgrt_sum = sum(filter(lambda x: x > 0, regret_sumsA[s].keys()))
    # #
    # #
    # #             #print("RSA: ", rgrt_sum)
    # #             if rgrt_sum > 0:
    # #                 piA[s][a] = (max(regret_sumsA[s][a], 0.0) / rgrt_sum)  # math.pow(t/(t+1), 2.0)
    # #             else:
    # #                 piA[s][a] = 1. / len(regret_sumsA[s].keys())  # * math.pow(t/(t+1), 2.0)
    # #
    # #             pi_sumsA[s][a] += piA[s][a] * gammaWeight # * math.pow(t/(t+1), 2.0) # Save the policy sums
    # #
    # #
    # #
    # #     # Update regret sums PLAYER 2
    # #     for s in sorted(QvaluesB.keys()):
    # #
    # #         target = 0.0
    # #         for a in total_actions:
    # #             target += QvaluesB[s][a] * piB[s][a]
    # #
    # #         for a in total_actions:
    # #             action_regret = QvaluesB[s][a] - target  # max(0.0, Qvalues1[s][a] - target)
    # #             if action_regret > 0:
    # #                 action_regret *= alphaWeight
    # #             else:
    # #                 action_regret *= betaWeight
    # #
    # #             regret_sumsB[s][a] += action_regret #max(0.0, action_regret) #action_regret
    # #
    # #     for s in sorted(piB.keys()):
    # #
    # #         rgrt_sum = 0.0
    # #         for a in total_actions:
    # #             rgrt_sum += max(0.0, regret_sumsB[s][a])
    # #
    # #         for a in total_actions:
    # #             #rgrt_sum = sum(filter(lambda x: x > 0, regret_sumsB[s].keys()))
    # #             #print("RSB: ", rgrt_sum)
    # #             if rgrt_sum > 0:
    # #                 piB[s][a] = (max(regret_sumsB[s][a], 0.) / rgrt_sum)  # math.pow(t/(t+1), 2.0)
    # #             else:
    # #                 piB[s][a] = 1.0 / len(regret_sumsB[s].keys())  # * math.pow(t/(t+1), 2.0)
    # #
    # #             pi_sumsB[s][a] += piB[s][a] * gammaWeight # * math.pow(t/(t+1), 2.0) # Save the policy sums
    #
    #
    #
    # print("Total valid start states: ", counter)
    #
    # # for s in total_states:
    # #     for a in total_actions:
    # #         print("S: ", s, "  A: ", actString(a), "  QA: ", lonrAgent.QValue_backupA[s][a], "  QB: ", lonrAgent.QValue_backupB[s][a])
######################################################################################






################################################
        #print(playerAStart, playerBStart)
        #playerAStart = playerAStart.de
    # for iii in range(10):
    #     print("--------------------")
    # player_a = Player(x=2, y=0, has_ball=False, p_id='A')
    # player_b = Player(x=1, y=0, has_ball=True, p_id='B')
    #
    # world = World()
    # world.set_world_size(x=cols, y=rows)
    # world.place_player(player_a, player_id='A')
    # world.place_player(player_b, player_id='B')
    # world.set_goals(100, 0, 'A')
    # world.set_goals(100, 3, 'B')
    # world.set_commentator_on()
    #
    # world.plot_grid()
    #
    # # print
    # # "actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4] \n"
    #
    # # print
    # # "Case where the player B scores an own goal:"
    # actions = {'A': 1, 'B': 1}
    # new_state, rewards, goal = world.move(actions)
    # print_status(goal, new_state, rewards, total_states)
    # world.plot_grid()
    #
    # actions = {'A': 4, 'B': 3}
    # new_state, rewards, goal = world.move(actions)
    # world.plot_grid()
    #
    #
    # print_status(goal, new_state, rewards, total_states)
    #
    # for iii in range(10):
    #     print("------------------------")
    # player_a = Player(x=2, y=0, has_ball=False, p_id='A')
    # player_b = Player(x=1, y=0, has_ball=True, p_id='B')
    #
    # # world = World()
    # world.place_player(player_a, player_id='A')
    # world.place_player(player_b, player_id='B')
    # world.plot_grid()
    # actions = {'A': 4, 'B': 1}
    # world.move(actions)
    # world.plot_grid()
    # actions = {'A': 4, 'B': 2}
    # #world.move(actions)
    # #new_state, rewards, goal = world.move(actions)
    # #world.plot_grid()
    #
    #
    # print_status(goal, new_state, rewards, total_states)


    # player_a = Player(x=2, y=0, has_ball=False, p_id='A')
    # player_b = Player(x=1, y=0, has_ball=True, p_id='B')
    #
    # world = World()
    # world.set_world_size(x=cols, y=rows)
    # world.place_player(player_a, player_id='A')
    # world.place_player(player_b, player_id='B')
    # world.set_goals(100, 0, 'A')
    # world.set_goals(100, 3, 'B')
    # world.set_commentator_on()
    #
    # world.plot_grid()
    #
    #
    # print("actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4]")
    #
    # actions = {'A': 1, 'B': 1}
    # new_state, rewards, goal = world.move(actions)
    # print_status(goal, new_state, rewards, total_states)
    # world.plot_grid()
    #
    # actions = {'A': 4, 'B': 3}
    # new_state, rewards, goal = world.move(actions)
    # world.plot_grid()
    #
    #
    # print_status(goal, new_state, rewards, total_states)


    ##############################################################
    ##############################################################
    # LEPS = 1
    #
    # for iters in range(LEPS):
    #     rows = 2
    #     cols = 4
    #     num_states = rows * cols
    #
    #     total_states = create_state_comb(list(range(num_states)), list(range(num_states)))
    #
    #     player_a = Player(x=2, y=0, has_ball=False, p_id='A')
    #     player_b = Player(x=1, y=0, has_ball=True, p_id='B')
    #
    #     world = World()
    #     world.set_world_size(x=cols, y=rows)
    #     world.place_player(player_a, player_id='A')
    #     world.place_player(player_b, player_id='B')
    #     world.set_goals(100, 0, 'A')
    #     world.set_goals(100, 3, 'B')
    #     world.set_commentator_on()
    #
    #     goal = False #True
    #
    #     playerAPos = int(str(player_a))
    #     playerBPos = int(str(player_b))
    #
    #     print("###############################")
    #     print("A: ", playerAPos)
    #     print("B: ", playerBPos)
    #
    #     cc = 0
    #     stateList = []
    #     while goal == False:
    #         #world.plot_grid()
    #
    #         stateListList = []
    #         # Get player positions
    #         # 0  1
    #         # 2  3
    #         #playerAPos = player_a.getCell() # int(str(player_a))
    #         #playerBPos = player_b.getCell() #int(str(player_b))
    #
    #
    #         # playerAPos = player_a.getCell(player_a.x, player_a.y)
    #         # playerBPos = player_b.getCell(player_b.x, player_b.y)
    #
    #         playerAPos = world.getCell("A") #player_a.getCell(player_a.x, player_a.y)
    #         playerBPos = world.getCell("B") #player_b.getCell(player_b.x, player_b.y)
    #
    #         print("PLAYER A POS: ", playerAPos)
    #
    #
    #         playerAAction = np.random.choice([0,1,2,3,4], p=piA[playerAPos])
    #         playerBAction = np.random.choice([0, 1, 2, 3, 4], p=piB[playerBPos])
    #
    #         stateListList.append(playerAPos)
    #         stateListList.append(playerBPos)
    #         stateListList.append(playerAAction)
    #         stateListList.append(playerBAction)
    #
    #         actions = {'A': playerAAction, 'B': playerBAction}
    #         new_state, rewards, goal = world.move(actions)
    #
    #         rewardA = rewards["A"]
    #         rewardB = rewards["B"]
    #
    #         stateListList.append(rewardA)
    #         stateListList.append(rewardB)
    #
    #         # newPlayerAPos = player_a.getCell()
    #         # newPlayerBPos = player_b.getCell()
    #
    #         # newPlayerAPos = player_a.getCell(player_a.x, player_a.y)
    #         # newPlayerBPos = player_b.getCell(player_b.x, player_b.y)
    #
    #         newPlayerAPos = world.getCell("A") #player_a.getCell(player_a.x, player_a.y)
    #         newPlayerBPos = world.getCell("B") #player_b.getCell(player_b.x, player_b.y)
    #
    #         print("NEW player A Pos: ", newPlayerAPos)
    #
    #         stateListList.append(newPlayerAPos)
    #         stateListList.append(newPlayerBPos)
    #
    #         stateListList.append(goal)
    #
    #         stateList.append(stateListList)
    #
    #     for ss in stateList:
    #         playerAPos, playerBPos, playerAAction, playerBAction, rewardA, rewardB, newPlayerAPos, newPlayerBPos, goal = ss
    #         print("PLAYER A POS : ", playerAPos)
    #
    #         alpha = 0.9999
    #         gamma = 0.9999
    #         # #if there is a goal
    #         if goal:
    #             # newPlayerAPos = player_a.getCell()
    #             # newPlayerBPos = player_b.getCell()
    #             # QValue_backupA[playerAPos][playerAAction] = rewardA
    #             # QValue_backupB[playerBPos][playerBAction] = rewardB
    #             Value = 0.0
    #             QValue_backupA[playerAPos][playerAAction] = QvaluesA[playerAPos][playerAAction] + alpha * (rewardA + gamma * Value - QvaluesA[playerAPos][playerAAction])
    #             QValue_backupB[playerBPos][playerBAction] = QvaluesB[playerBPos][playerBAction] + alpha * (rewardB + gamma * Value - QvaluesB[playerBPos][playerBAction])
    #
    #
    #
    #         else:
    #
    #             #Qupdate for Player A
    #             #rewardA = 0.0
    #             #newPlayerAPos = player_a.getCell()# int(str(player_a))
    #
    #             Value = 0.0
    #             for action_ in range(5):
    #                 Value += QvaluesA[newPlayerAPos][action_] * piA[newPlayerAPos][action_]
    #
    #             QValue_backupA[playerAPos][playerAAction] = QvaluesA[playerAPos][playerAAction] + alpha * (rewardA + gamma * Value - QvaluesA[playerAPos][playerAAction])
    #             #Qvalues_backup[s][a_ID] = (self.Qvalues[s][a_ID] + self.alpha * (reward + self.gamma * Value - self.Qvalues[s][a_ID]))
    #
    #
    #                 # QUpdate for Player B
    #             #rewardB = 0.0
    #             #newPlayerBPos = player_b.getCell() # int(str(player_b))
    #
    #             Value = 0.0
    #             for action_ in range(5):
    #                 Value += QvaluesB[newPlayerBPos][action_] * piB[newPlayerBPos][action_]
    #
    #             QValue_backupB[playerBPos][playerBAction] = QvaluesB[playerBPos][playerBAction] + alpha * (rewardB + gamma * Value - QvaluesB[playerBPos][playerBAction])
    #
    #
    #
    #         for ss in range(state_size):
    #             for aa in range(n_actions):
    #                 QvaluesA[ss][aa] = QValue_backupA[ss][aa]
    #                 QvaluesB[ss][aa] = QValue_backupB[ss][aa]
    #         #print("Rewards: ", rewards)
    #
    #         # cc += 1
    #         # if cc > 10:
    #         #     print("WEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
    #         #     goal = True
    #         #print_status(goal, new_state, rewards, total_states)
    #         #world.plot_grid()
    #         # print("_________________")
    #         # print("new_state: ", new_state)
    #         # print("GOAL: ", goal)
    #         # print("_________________")
    #
    # print("")
    # print("")
    # print("Player A QValues: ")
    # print(QvaluesA)
    # print("")
    # print("Player B QValues: ")
    # print(QvaluesB)
    # print("")
    # plA = QvaluesA[2][1]  #player A starting position, going south
    # plB = QvaluesB[1][4]
    # print("HERE: ")
    # print("   A", plA)
    # print("   B", plB)
    # print("DONE")
    #
    #############################################################
    #############################################################

    # actions = {'A': 1, 'B': 2}
    # new_state, rewards, goal = world.move(actions)
    # print_status(goal, new_state, rewards, total_states)
    # world.plot_grid()
    # print("GOAL: ", goal)
    #
    # actions = {'A': 1, 'B': 2}
    # new_state, rewards, goal = world.move(actions)
    # print_status(goal, new_state, rewards, total_states)
    # world.plot_grid()
    # print("GOAL: ", goal)

    # world.plot_grid()
    #
    # print("actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4] \n")
    #
    # print ("Case where the player B scores an own goal:")
    # actions = {'A': 1, 'B': 1}
    # new_state, rewards, goal = world.move(actions)
    # print_status(goal, new_state, rewards, total_states)
    # world.plot_grid()
    #
    # actions = {'A': 4, 'B': 3}
    # new_state, rewards, goal = world.move(actions)
    # world.plot_grid()
    # print_status(goal, new_state, rewards, total_states)
    #
    # print("Case where player B scores while A doesn't move:")
    # # After a goal the environment needs to be reset
    # world.place_player(player_a, player_id='A')
    # world.place_player(player_b, player_id='B')
    # actions = {'A': 4, 'B': 1}
    # world.move(actions)
    # actions = {'A': 4, 'B': 2}
    # world.move(actions)
    # new_state, rewards, goal = world.move(actions)
    # world.plot_grid()
    # print_status(goal, new_state, rewards, total_states)
    #
    # print("Case where the two player collide:")
    # # After a goal the environment needs to be reset
    # world.place_player(player_a, player_id='A')
    # world.place_player(player_b, player_id='B')
    # actions = {'A': 4, 'B': 2}
    # new_state, rewards, goal = world.move(actions)
    # world.plot_grid()
    # print_status(goal, new_state, rewards, total_states)


if __name__ == '__main__':
    main()
