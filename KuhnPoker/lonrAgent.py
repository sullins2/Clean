from KuhnPoker.misc import *
import numpy as np

import copy

class Agent:
    def __init__(self, totState=None, infosets=None, totInfoStates=None):


        self.totState = totState
        self.infosets = infosets
        self.totInfoStates = totInfoStates

        self.state_size = len(self.totState)

        #self.alpha = 0.9999
        self.gamma = 0.9999


        # Various for information sets
        self.sigma = init_strategy(self.totInfoStates[0])  # init strategies to uniform
        self.QvaluesINF = init_empty_node_maps(self.totInfoStates[0])
        self.Qvalues_backupINF = init_empty_node_maps(self.totInfoStates[0])
        self.Qvalues_sumsINF = init_empty_node_maps(self.totInfoStates[0])
        self.regret_sumsINF = init_empty_node_maps(self.totInfoStates[0])  # zero-init regret sums at each node
        self.sigma_sums = init_empty_node_maps(self.totInfoStates[0])
        self.nash_equilibrium = init_empty_node_maps(self.totInfoStates[0])



        # Q values for states
        self.Qvalues = np.zeros((self.state_size, 2))
        self.Qvalues_backup = np.zeros((self.state_size, 2))
        self.Qvalues_sums = np.zeros((self.state_size, 2))
        self.sigmaLOC = np.zeros((self.state_size, 2))

        self.mapToINF = {}

        RANDOMIZE = False
        if RANDOMIZE:
            for k in self.sigma.keys():
                for a in self.sigma[k]:
                    self.sigma[k][a] = 1.0 / float(len(list(self.sigma[k].keys())))



    def simulate_lonr(self, t=0, T=100, player=0):







        LOG = False



        for s in range(self.state_size):

            self.mapToINF[s] = str(self.totState[s])

            if LOG:
                print("Just entered start of main loop through statts. s = ", s)

            # Skipping over root chance state.
            if self.totState[s].is_chance():
                if LOG:
                    print(" - state is chance, continuing. s = ", s)
                continue

            # Skip terminal
            if self.totState[s].is_terminal():
                if LOG:
                    print(" - state is terminal, continuing. s = ", s)
                continue

            # Get the info set of this state
            state_info_set = self.totState[s].inf_set()
            if LOG:
                print("State info set (s): ", state_info_set)

            # Loop through every action in the current state
            if LOG:
                print("Now going to loop thru every action in state s = ", s)
            for a in self.totState[s].actions:



                next_state = self.totState[s].play(a)
                next_state_id = self.totState.index(next_state)
                next_state_inf_set = next_state.inf_set()

                # ORIG: .J.
                # NEXT_STATE: .Q.BET action was BET
                if LOG:
                    print(" - action: ", a, "  next_state_id: ", next_state_id, " next state info set: ", next_state_inf_set)

                a_ID = self.totState[s].actions.index(a)

                reward = 0.0

                next_player = self.totState[s].next_player
                if self.totState[s].is_chance():
                    next_player = 1.0
                # next_player = -next_player

                #inf_set = next_state_inf_set
                if self.totState[next_state_id].is_terminal():
                    reward = self.totState[next_state_id].evaluation()
                    if LOG:
                        print(" - next_state is TERMINAL. reward: ", reward)

                if LOG:
                    print(" - About to loop through actions of next_state_id: ", next_state_id)
                Value = 0.0

                # Looping thru .Q.BET actions which means you have a Q and there was a BET by first player
                # The actions available are CALL, FOLD
                for action_ in self.totState[next_state_id].actions:
                    act_ID = self.totState[next_state_id].actions.index(action_)
                    if LOG:
                        print("  - next_state_inf_set: ", next_state_inf_set, "  action_: ", action_)
                    # Value += Qvalue[.Q.BET][CALL] * sigma[.Q.BET][CALL] + Qvalue[.Q.BET][FOLD] + Qvalue[.Q.BET][FOLD]



                    Value += self.Qvalues[next_state_id][act_ID] * self.sigma[next_state_inf_set][action_]# * next_player# * 0.5 # ** self.sigma[state_info_set][a]

                #
                if LOG:
                    print(" - QUpdate. s = ", s, "  a_ID: ", a_ID, " state_info_set: ", state_info_set, "  a: ", a)
                self.alpha = 0.99
                self.Qvalues_backup[s][a_ID] = (self.Qvalues[s][a_ID] + self.alpha * (reward + self.gamma * Value - self.Qvalues[s][a_ID]))
                #self.Qvalues_backup[s][a_ID] = (self.QvaluesINF[state_info_set][a] + self.alpha * (reward + self.gamma * Value - self.QvaluesINF[state_info_set][a]))
                #self.Qvalues_backupINF[state_info_set][a] += (self.QvaluesINF[state_info_set][a] + self.alpha * (reward + self.gamma * Value - self.QvaluesINF[state_info_set][a]))


        #self.QvaluesINF = copy.deepcopy(self.Qvalues_backupINF)

        # Copy QValues over, reset QValues for info_sets
        self.Qvalues = copy.deepcopy(self.Qvalues_backup)
        #self.QvaluesINF = misc.init_empty_node_maps(self.totInfoStates[0])

        self._cfr_utility_recursive(self.totState[0], 1.0, 1.0)