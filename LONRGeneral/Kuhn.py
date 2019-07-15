from LONRGeneral.misc import *
import random
from LONRGeneral.LONR import *
from copy import  *

class Agent:
    def __init__(self, totState=None, infosets=None, totInfoStates=None):


        self.totState = totState
        self.infosets = infosets
        self.totInfoStates = totInfoStates

        self.state_size = len(self.totState)

        self.alpha = 0.9999
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
                    self.sigma[k][a] = 1.0 / float(np.random.randint(1, 100))




    def updateReachProbs(self, node, reach_0, reach_1):

        #print("INFSET: ", node.inf_set())
        if node.is_terminal():
            #print("TERM : ", node)
            #HADnode.reach_prob = reach_0 * reach_1
            #child_reach_0 = reach_0 * (1.0 if node.next_player == misc.player0 else 1)
            #child_reach_1 = reach_1 * (1.0 if node.next_player == -misc.player0 else 1)

            node.reach_prob = reach_0 if node.next_player ==player0 else reach_1
            node.reach_prob0 = reach_0
            node.reach_prob1 = reach_1
            return #1.0 #node.reach_prob

        elif node.is_chance():
            #print("CHANCE: ", node)
            chance_outcomes = {node.play(action) for action in node.actions}
            node.reach_prob = 1.0 / 6.0#node.chance_prob()
            node.reach_prob0 = 1.0 / 6.0
            node.reach_prob1 = 1.0 / 6.0
            for outcome in chance_outcomes:
                self.updateReachProbs(outcome, 1.0 / 6.0, 1.0 / 6.0)


        elif node.is_chance() == False:
            for action in node.actions:

                child_reach_0 = reach_0 * (self.sigma[node.inf_set()][action] if node.next_player == player0 else 1)
                child_reach_1 = reach_1 * (self.sigma[node.inf_set()][action] if node.next_player == -player0 else 1)


                node.reach_prob = reach_0 if node.next_player == player0 else reach_1
                node.reach_prob0 = reach_0
                node.reach_prob1 = reach_1
                self.updateReachProbs(node.play(action), child_reach_0, child_reach_1)






    def simulate_lonr(self, t=0, T=100, RM=True, player=0):

        if RM == False:
            for ii in range(20):
                print("Error: RM and MW both false!")

        #self.alpha = self.alpha * 0.99
        # Update all the reach probabilities

        # root = self.totState[0]
        # self.updateReachProbs(root, 1.0 / 1.0, 1.0 / 1.0)

        #self.alpha *= 0.9999

        LOG = False



        for s in range(self.state_size):
        # for i in range(1):
        #     s = np.random.randint(0, len(self.totState))

            # if self.totState[s].next_player != player:
            #     continue

            self.mapToINF[s] = str(self.totState[s])

            #print("STATE: ", str(self.totState[s]), " SIZE: ", len(self.totState))
            if LOG:
                print("Just entered start of main loop through statts. s = ", s)

            # Skipping over root chance state.
            if self.totState[s].is_chance():
                if LOG:
                    print(" - state is chance, continuing. s = ", s)
                continue

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
            # for ii in range(1):
            #     aa = np.random.randint(0,2)
            #     c = 0
            #     a = None
            #     for aaa in self.totState[s].actions:
            #         if aa == c:
            #             a = aaa
            #         c += 1


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
                    Value += self.Qvalues[next_state_id][act_ID] * self.sigma[next_state_inf_set][action_]# * next_player# * 0.5 # ** self.sigma[state_info_set][a]

                #
                if LOG:
                    print(" - QUpdate. s = ", s, "  a_ID: ", a_ID, " state_info_set: ", state_info_set, "  a: ", a)
                self.Qvalues_backup[s][a_ID] = (self.Qvalues[s][a_ID] + self.alpha * (reward + self.gamma * Value - self.Qvalues[s][a_ID]))


        self.Qvalues = deepcopy(self.Qvalues_backup)
        #self.QvaluesINF = misc.init_empty_node_maps(self.totInfoStates[0])

        self._cfr_utility_recursive(self.totState[0], 1.0, 1.0)

    def _cfr_utility_recursive(self, state, reach_0, reach_1):

        # print("State: ", state.inf_set(), "  REACH0: ", reach_0, " REACH1: ", reach_1)
        children_states_utilities = {}  # map from child_state[action] => utility of child state

        # base case: return value at terminal node
        if state.is_terminal():
            # evaluate terminal node according to the game result
            return  # state.evaluation()

        # chance node
        chance_sampling = False
        if state.is_chance():
            if chance_sampling:  # self.chance_sampling:
                # if node is a chance node, sample one child node and proceed normally
                chance_outcomes = [action for action in state.actions]
                rn = np.random.randint(0, len(chance_outcomes))
                return self._cfr_utility_recursive(state.play(chance_outcomes[rn]), reach_0 * (1.0 / 6.0), reach_1 * (1.0 / 6.0))  # state.sample_one()
            else:
                # If not chance_sampling, all outcomes considered

                # state.play(action) returns child node
                # get all chance outcomes of children
                # chance_outcomes = {state.play(action) for action in state.actions}

                # recursive call
                # return chance of going to the child state TIMES overall sum of utility for all children states
                # - that is the (recursively generated) utility at the chance node
                # this is an expansive call, but remember it is only made at single chance node at root
                # return state.chance_prob() * sum([self._cfr_utility_recursive(outcome, reach_0, reach_1) for outcome in chance_outcomes])

                for action in state.actions:
                    outcome = state.play(action)
                    self._cfr_utility_recursive(outcome, reach_0 * (1.0 / 6.0), reach_1 * (1.0 / 6.0))

            return
        # If this point is gotten to, then node is a MOVE state (NOT chance or terminal)
        # Note that this is the "middle" of game tree (top is single CHANCE node, bottom is all terminal nodes)

        # sum up all utilities for playing actions in our game state
        value = 0.0
        state_id = self.totState.index(state)
        for action in state.actions:

            child_reach_0 = reach_0 * (self.sigma[state.inf_set()][action] if state.next_player == 1 else 1)
            child_reach_1 = reach_1 * (self.sigma[state.inf_set()][action] if state.next_player == -1 else 1)


            a_ID = state.actions.index(action)


            child_state_utility = self.Qvalues[state_id][a_ID]

            value += self.sigma[state.inf_set()][action] * self.Qvalues[state_id][a_ID]

            # values for chosen actions (child nodes) are kept here
            children_states_utilities[action] = child_state_utility


        (cfr_reach, reach) = (reach_1, reach_0) if state.next_player == 1 else (reach_0, reach_1)

        # Updates for sum of regrets and sum of sigmas (strategy)
        for action in state.actions:
            action_cfr_regret = state.next_player * cfr_reach * (children_states_utilities[action] - value)

            self.regret_sumsINF[state.inf_set()][action] += action_cfr_regret

            self.sigma_sums[state.inf_set()][action] += self.sigma[state.inf_set()][action] * reach

        state_id = self.totState.index(state)
        for action in state.actions:
            child_reach_0 = reach_0 * (self.sigma[state.inf_set()][action] if state.next_player == 1 else 1)
            child_reach_1 = reach_1 * (self.sigma[state.inf_set()][action] if state.next_player == -1 else 1)
            self._cfr_utility_recursive(state.play(action), child_reach_0, child_reach_1)

        # If chance_sampling, update strategy
        # NOTE: If VanillaCFR: _update_sigma() is called from root after each training iteration
        # if self.chance_sampling:
        # update sigma according to cumulative regrets - we can do it here because we are using chance sampling
        # and so we only visit single game_state from an information set (chance is sampled once)
        #   self._update_sigma(state.inf_set())

        return

    def compute_nash_equilibrium(self):
        self.__compute_ne_rec(self.totState[0])

    def __compute_ne_rec(self, node):
        if node.is_terminal():
            return  # node.evaluation()
        i = node.inf_set()
        if node.is_chance():

            self.nash_equilibrium[i] = {a: node.chance_prob() for a in node.actions}
        else:
            # sigma_sum = max(1.0, sum(self.sigma_sums[i].values()))
            sigma_sum = sum(self.sigma_sums[i].values())
            self.nash_equilibrium[i] = {a: self.sigma_sums[i][a] / sigma_sum for a in node.actions}
        # go to subtrees
        for k in node.children:
            self.__compute_ne_rec(node.children[k])

    def _update_sigma_recursively(self):
        for s in range(len(self.totInfoStates)):

            state_info_set = self.totInfoStates[s].inf_set()

            # inf_set = state_info_set
            rgrt_sum = sum(filter(lambda x: x > 0, self.regret_sumsINF[state_info_set].values()))

            # Update sigma with positive regrets or uniform
            for a in self.regret_sumsINF[state_info_set].keys():
                # if state_info_set == '.K.':
                #     print("K: regret sum: ", rgrt_sum, " a: ", a, "  sigma: ", self.sigma[state_info_set][a])
                self.sigma[state_info_set][a] = max(self.regret_sumsINF[state_info_set][a],0.) / rgrt_sum if rgrt_sum > 0 else 1. / len(self.regret_sumsINF[state_info_set].keys())
                # if state_info_set == '.K.':
                #     print("  sigma got set to: ", self.sigma[state_info_set][a])

    # Add regret to information set for action
    def _add_cfr_regret(self, information_set, action, regret):
        self.regret_sumsINF[information_set][action] += regret

    # Add probability to sigma sums
    def _add_sigma(self, information_set, action, prob):
        self.sigma_sums[information_set][action] += prob

    def game_value(self):
        return self.__game_value_recursive(self.totState[0])
        # return self.__game_value_recursive(self.totInfoStates[0])

    def __game_value_recursive(self, node):
        value = 0.
        if node.is_terminal():
            return node.evaluation()
        for action in node.actions:
            value += self.nash_equilibrium[node.inf_set()][action] * self.__game_value_recursive(node.play(action))

        return value


def train_lonr(agent, n_episodes=0, T=0, RM=True, MW=False, CFR_PLUS=False, WEIGHTED_REGRETS=False):
    # print("Starting lonr agent training..")

    for ep in range(1, n_episodes + 1):
        # r_sum = agent.simulate_lonr()
        if ep % 1000 == 0:
            print("Iteration: ", ep)
        agent.simulate_lonr(t=ep, T=T, RM=RM, player=0)
        agent._update_sigma_recursively()






####################
# 2 player Kuhn Poker Game Tree
####################

totState = []

totInfoStates = []

infosets = []

# Base class for a game state
class KuhnBase:

    def __init__(self, parent, next_player, actions):
        self.parent = parent            # parent node
        self.next_player = next_player  # which player (0 or 1) has next action
        self.actions = actions          # available actions at this node
        self.reach_prob = 1.0
        self.reach_prob0 = 1.0
        self.reach_prob1 = 1.0


    # Simulate playing an action
    # Returns the child node of the given action
    def play(self, action):
        return self.children[action]

    def is_chance(self):
        return self.next_player == CHANCE   # Check if chance node

    #@property
    def inf_set(self):
        raise NotImplementedError("Please implement information_set method")


# Chance game state at root of tree
# called as:
#           root = KuhnRootChanceGameState(actions=deck)


class KuhnRootChanceGameState(KuhnBase):

    def __init__(self, actions):
        super().__init__(parent = None, next_player = CHANCE, actions = actions)

        self.cards = []
        self.actions_history = "."


        totState.append(self)
        totInfoStates.append(self)

        # map of children
        # cards (actions) => game states
        # first player to act is player0, history is empty, each card pair, first action choices are CHECK or BET
        #print(self.actions)
        self.children = {
            cards: KuhnPlayerMoveGameState(
                self, player0, [], cards, [BET, CHECK]
            ) for cards in self.actions
        }
        self._chance_prob = 1.0 / 6.0#len(self.children)
        self.reach_prob = 1.0 / 6.0#self._chance_prob
        # Chance probability is uniform among number of children
        #self._chance_prob = 1.0 / float(len(self.children))


    # Always false, only chance node is at root
    def is_terminal(self):
        return False

    def inf_set(self):
        return "."   #history string (infoset), empty at root

    def chance_prob(self):
        return self._chance_prob

    def sample_one(self):
        return random.choice(list(self.children.values()))


# Game states that make up the game tree
# These make up entire tree, except for root which is CHANCE
class KuhnPlayerMoveGameState(KuhnBase):

    def __init__(self, parent, next_player, actions_history, cards, actions):
        super().__init__(parent = parent, next_player = next_player, actions = actions)

        self.actions_history = actions_history  # string of history up to this node
        self.cards = cards                      # cards players have

        # infoset is CARD + history
        public_card = self.cards[0] if self.next_player == player0 else self.cards[1]
        self._information_set = ".{0}.{1}".format(public_card, ".".join(self.actions_history))

        totState.append(self)
        if self._information_set not in infosets:
            infosets.append(self._information_set)
            totInfoStates.append(self)

        # Map of children
        # - actions => child game states
        self.children = {
            a : KuhnPlayerMoveGameState(
                self,
                -next_player,                       # switch player turns in next state
                self.actions_history + [a],           # append action to history
                cards,                              # propagate cards down tree
                self.__get_actions_in_next_round(a) # pass valid actions for child node
            ) for a in self.actions
        }

        #


    # legal actions depending on what actions have been taken
    def __get_actions_in_next_round(self, a):
        if len(self.actions_history) == 0 and a == BET:         #first move BET => FOLD or CALL
            return [CALL, FOLD]#[FOLD, CALL]
        elif len(self.actions_history) == 0 and a == CHECK:     #first move CHECK => BET or CHECK
            return [BET, CHECK]
        elif self.actions_history[-1] == CHECK and a == BET:    # first move CHECK, next action is BET => CALL or FOLD
            return [CALL, FOLD]
        # The rest are all terminal so no moves are left
        elif a == CALL or a == FOLD or (self.actions_history[-1] == CHECK and a == CHECK):
            return []

    def inf_set(self):
        return self._information_set

    def is_terminal(self):
        return self.actions == []

    def evaluation(self):
        if self.is_terminal() == False:
            raise RuntimeError("trying to evaluate non-terminal node")

        if self.actions_history[-1] == CHECK and self.actions_history[-2] == CHECK:
            return winMap[self.cards] * 1 # only ante is won/lost

        if self.actions_history[-2] == BET and self.actions_history[-1] == CALL:
            return winMap[self.cards] * 2 #Bet and a call, so 2 chips each in pot

        if self.actions_history[-2] == BET and self.actions_history[-1] == FOLD:
            return self.next_player * 1

        print("MISSING AN EVALUATION")