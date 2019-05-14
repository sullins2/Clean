
import numpy as np


class TigerAgent:
    """ LONR Agent for Tiger Game

    """


    def __init__(self, tigergame):

        # Save Tiger game
        self.tigergame = tigergame

        # Parameters
        self.gamma = 0.59

        # Q-Values
        self.Q = {}
        self.Q_bu = {}
        self.Qsums = {}

        # Policies
        self.pi = {}
        self.pi_sums = {}

        # Regret Sums
        self.regret_sums = {}

        # Show prints
        self.VERBOSE = False

        self.TURN = 0

        # Initialize the states and info sets
        # Initialize policy to be uniform over total actions
        self.tot = self.tigergame.totalStatesRight + self.tigergame.totalStatesLeft
        self.tot = self.tigergame.totalStates
        for s in self.tot:#self.tigergame.totalStates:

            for a in self.tigergame.getActions(s):

                # Add entry into Qs
                if s not in self.Q:
                    self.Q[s] = {}
                    self.Q_bu[s] = {}
                    self.Qsums[s] = {}
                self.Q[s][a] = 0.0#self.tigergame.getReward(s,a)
                self.Q_bu[s][a] = 0.0# self.tigergame.getReward(s,a)
                self.Qsums[s][a] = 0.0

                stateInfoSet = self.tigergame.stateIDtoIS(s)
                if stateInfoSet == None: continue
                if stateInfoSet not in self.pi:
                    self.pi[stateInfoSet] = {}
                    self.pi_sums[stateInfoSet] = {}
                    self.regret_sums[stateInfoSet] = {}

                self.pi[stateInfoSet][a] = 1.0 / len(self.tigergame.getActions(stateInfoSet))
                self.pi_sums[stateInfoSet][a] = 0.0
                self.regret_sums[stateInfoSet][a] = 0.0



    def train_lonr_value_iteration(self, iterations=0):
        print("Starting lonr agent training..")
        for i in range(1, iterations + 1):

            self._train_lonr_value_iteration(t=i)

            if True:
                if i % 1000 == 0 and i > 0:
                    print("Iteration:", i)
        print("Finished LONR Training.")


    def _train_lonr_value_iteration(self, t=0):
        """ One complete value iteration

        """

        rand = np.random.randint(0, 100)
        #if self.TURN == 0:
        if rand < 61:
            #print(rand)
            stateList = self.tigergame.totalStatesLeft
            self.TURN = 1
        else:
            #print(rand)
            stateList = self.tigergame.totalStatesRight
            self.TURN = 0

        #stateList = self.tigergame.totalStates

        for s in stateList: #self.tigergame.totalStates:

            # if s == "root":
            #     continue

            self.verbose("Just entered overall state loop s: ", s)
            # Terminal - Set Q as Reward (no actions in terminal states)
            # - alternatively, one action which is 'exit' which gets full reward
            if self.tigergame.isTerminal(s):
                for a in self.tigergame.getActions(s):
                    self.verbose(" - Terminal setting: s: ", s, " a: ", a, "  rew: ", self.tigergame.getReward(s))
                    self.Q_bu[s][a] = self.tigergame.getReward(s)
                continue


            # Loop through all actions in current state s
            for a in self.tigergame.getActions(s):

                self.verbose(" - action: ", a)

                # Get successor states
                succs = self.tigergame.getNextStatesAndProbs(s, a)

                self.verbose(" - succs: ", succs)

                # if s == "rootTLLGL" and a == self.tigergame.OPENLEFT:
                #     print("HERE")
                #     print(self.tigergame.getReward(s, a=a))
                #     print(s, a)

                Value = 0.0
                for _, s_prime, prob in succs:
                    tempValue = 0.0
                    self.verbose("   - loop: s_prime: ", s_prime, "  prob:", prob)
                    for a_prime in self.tigergame.getActions(s_prime):
                        tempValue += self.Q[s_prime][a_prime] * self.pi[self.tigergame.stateIDtoIS(s_prime)][a_prime]
                    self.verbose("  - afterLoop reward s: ", s, "  reward: ", self.tigergame.getReward(s, a=a))
                    #Value += prob * (self.tigergame.getReward(s, a=a) + self.gamma * tempValue)
                    Value += prob * (self.tigergame.getReward(s, a=a) + self.gamma * tempValue)
                self.Q_bu[s][a] = self.gamma * Value


        # Copy over Q Values
        for s in self.tigergame.totalStates:
            for a in self.tigergame.getActions(s):
                self.Q[s][a] = self.Q_bu[s][a]
                self.Qsums[s][a] += self.Q[s][a]


        # Regrets and policy updates


        # Update regret sums
        for s in stateList: #self.tigergame.totalStates:

            # Skip terminals
            if self.tigergame.isTerminal(s):
                continue

            # Skip updating Tiger location probability
            if s == "root":
                continue

            target = 0.0

            for a in self.tigergame.getActions(s):
                target += self.Q[s][a] * self.pi[self.tigergame.stateIDtoIS(s)][a]

            for a in self.tigergame.getActions(s):
                action_regret = self.Q[s][a] - target

                iters = t
                alphaR = 3.0 / 2.0  # accum pos regrets
                betaR = 0.0  # accum neg regrets


                alphaW = pow(iters, alphaR)
                alphaWeight = (alphaW / (alphaW + 1))

                betaW = pow(iters, betaR)
                betaWeight = (betaW / (betaW + 1))



                if action_regret > 0:
                    action_regret *= alphaWeight
                else:
                    action_regret *= betaWeight

                RMP = True
                if RMP:
                    self.regret_sums[self.tigergame.stateIDtoIS(s)][a] = max(0.0, self.regret_sums[self.tigergame.stateIDtoIS(s)][a] + action_regret)
                else:
                    self.regret_sums[self.tigergame.stateIDtoIS(s)][a] += action_regret # max(0.0, self.regret_sums[self.tigergame.stateIDtoIS(s)][a] + action_regret)

        # Regret Match
        for s in stateList: # self.tigergame.totalStates:

            if self.tigergame.isTerminal(s):
                continue

            if s == "root":
                continue

            for a in self.tigergame.getActions(s):
                rgrt_sum = 0.0
                for aa in self.regret_sums[self.tigergame.stateIDtoIS(s)].keys():
                    rgrt_sum += self.regret_sums[self.tigergame.stateIDtoIS(s)][aa] if self.regret_sums[self.tigergame.stateIDtoIS(s)][aa] > 0 else 0.0

                # Check if this is a "trick"
                # Check if this can go here or not
                if rgrt_sum > 0:
                    self.pi[self.tigergame.stateIDtoIS(s)][a] = (max(self.regret_sums[self.tigergame.stateIDtoIS(s)][a], 0.) / rgrt_sum)
                else:
                    self.pi[self.tigergame.stateIDtoIS(s)][a] = 1.0 / len(self.regret_sums[self.tigergame.stateIDtoIS(s)])

                # Does this go above or below
                gammaR = 2.0  # contribution to avg strategy
                gammaWeight = pow((t / (t + 1)), gammaR)
                self.pi_sums[self.tigergame.stateIDtoIS(s)][a] += self.pi[self.tigergame.stateIDtoIS(s)][a] * gammaWeight


    def verbose(self, *args):
        if self.VERBOSE:
            for arg in args:
                print(arg, end='')
            print('')

class TigerGame(object):

    def __init__(self):

        # Actions
        self.OPENLEFT = "OL"
        self.LISTEN = "L"
        self.OPENRIGHT = "OR"

        self.TL = "TL"
        self.TR = "TR"

        #total action list
        self.totalActions = [self.OPENLEFT, self.LISTEN, self.OPENRIGHT]

        # State IDs
        # depth = 0
        # root goes 50/50 to each MDP where Tiger is set
        self.root = "root"

        # depth = 1
        self.rootTL = "rootTL"     # MDP with Tiger on left
        self.rootTR = "rootTR"     # MDP with Tiger on right

        # depth = 2
        self.rootTLOL = "rootTLOL"       # Tiger on LEFT, OL
        self.rootTLOR = "rootTLOR"       # Tiger on LEFT, OR

        self.rootTROL = "rootTROL"       # Tiger on RIGHT, OL
        self.rootTROR = "rootTROR"       # Tiger on RIGHT, OR

        # depth = 3
        self.rootTLLGL = "rootTLLGL"      # Tiger on LEFT, LISTEN, GL
        self.rootTLLGR = "rootTLLGR"     # Tiger on LEFT, LISTEN, GR

        self.rootTRLGL = "rootTRLGL"     # Tiger on RIGHT, LISTEN, GL
        self.rootTRLGR = "rootTRLGR"     # Tiger on RIGHT, LISTEN, GR

        # depth = 4
        self.rootTLLGLLGL = "rootTLLGLLGL" # Tiger on left, Listen, Growl Left, Listen, Growl Left
        self.rootTLLGLLGR = "rootTLLGLLGR"
        self.rootTLLGRLGL = "rootTLLGRLGL"
        self.rootTLLGRLGR = "rootTLLGRLGR"

        self.rootTRLGLLGL = "rootTRLGLLGL" # Tiger on right,
        self.rootTRLGLLGR = "rootTRLGLLGR"
        self.rootTRLGRLGL = "rootTRLGRLGL"
        self.rootTRLGRLGR = "rootTRLGRLGR"

        self.totalStates = [self.root,
                            self.rootTL, self.rootTR,
                            self.rootTLOL, self.rootTLOR, self.rootTROL, self.rootTROR,
                            self.rootTLLGL, self.rootTLLGR, self.rootTRLGL, self.rootTRLGR,
                            self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR,
                            self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR]

        self.totalStatesLeft = [self.rootTL,
                                self.rootTLOL, self.rootTLOR,
                                self.rootTLLGL, self.rootTLLGR,
                                self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR]

        self.totalStatesRight = [self.rootTR,
                                 self.rootTROL, self.rootTROR,
                                 self.rootTRLGL, self.rootTRLGR,
                                 self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR]




    def stateIDtoIS(self, state):

        # use pi, regret, etc for LEFT
        if state == self.rootTL or state == self.rootTR:
            return self.rootTL
        elif state == self.rootTLLGL or state == self.rootTRLGL:
            return self.rootTLLGL
        elif state == self.rootTLLGR or state == self.rootTRLGR:
            return self.rootTLLGR
        elif state == self.rootTLLGLLGL or state == self.rootTRLGLLGL:
            return self.rootTLLGLLGL
        elif state == self.rootTLLGLLGR or state == self.rootTRLGLLGR:
            return self.rootTLLGLLGR
        elif state == self.rootTLLGRLGL or state == self.rootTRLGRLGL:
            return self.rootTLLGRLGL
        elif state == self.rootTLLGRLGR or state == self.rootTRLGRLGR:
            return self.rootTLLGRLGR

        #TODO - add depth 4 ones

        elif state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
            return state
        elif state == self.root:
            return state
        else:
            return None

    def getActions(self, state):
        if state == self.root:
            return [self.TL, self.TR]
        elif state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
            return ["exit"]
        else:
            return [self.OPENLEFT, self.LISTEN, self.OPENRIGHT]

    def getReward(self, state, a = None):
        # Tiger left, open LEFT
        if state == self.root:
            return 0.0
        elif state == self.rootTL or state == self.rootTR:
            if a == self.LISTEN:
                return -1.0
            else:
                return 0.0
        if state == self.rootTLOL:
            return -100.0
        # Tiger left, open RIGHT
        elif state == self.rootTLOR:
            return 10.0
        # Tiger right, open LEFT
        elif state == self.rootTROL:
            return 10.0
        # Tiger right, open RIGHT
        elif state == self.rootTROR:
            return -100.0
        # Everywhere else is a listen
        elif state == self.rootTLLGL or state == self.rootTLLGR or state == self.rootTRLGL or state == self.rootTRLGR:
            if a == self.LISTEN:
                return -1.0
            else:
                return 0.0
        elif state == self.rootTLLGLLGL or state == self.rootTLLGLLGR or state == self.rootTLLGRLGL or state == self.rootTLLGRLGR:
            if a == self.LISTEN:
                return -1.0
            else:
                return 0.0
        elif state == self.rootTRLGLLGL or state == self.rootTRLGLLGR or state == self.rootTRLGRLGL or state == self.rootTRLGRLGR:
            if a == self.LISTEN:
                return -1.0
            else:
                return 0.0


        else:
            print("Not caught: ", state)
            return 0.0
            # return self.getReward(state, a)

    def isTerminal(self, state):
        if state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
            return True
        else:
            return False


    def getNextStatesAndProbs(self, state, action):

        #self.UP, upState, 1.0 - self.noise]

        # Top most root
        if state == self.root:
            if action == "TL":
                return [[None, self.rootTL, 0.5]]
            else:
                return [[None, self.rootTR, 0.5]]

        # Top root of Tiger on left
        elif state == self.rootTL:

            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGL, 0.85], [None, self.rootTLLGR, 0.15]]

        # Top root of Tiger on right
        elif state == self.rootTR:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGL, 0.15], [None, self.rootTRLGR, 0.85]]

        # Tiger on left, Listen, GL
        elif state == self.rootTLLGL:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGLLGL, 0.85], [None, self.rootTLLGLLGR, 0.15]]

        # Tiger on left, LISTEN, GR
        elif state == self.rootTLLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGRLGL, 0.85], [None, self.rootTLLGRLGR, 0.15]]

        # Tiger on RIGHT, Listen, GL
        elif state == self.rootTRLGL:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGLLGL, 0.15], [None, self.rootTRLGLLGR, 0.85]]

        # Tiger on RIGHT, LISTEN, GR
        elif state == self.rootTRLGR:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGRLGL, 0.15], [None, self.rootTRLGRLGR, 0.85]]

    ## Bottom depth
        # Tiger on left
        elif state == self.rootTLLGLLGL:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGLLGL, 0.85], [None, self.rootTLLGLLGR, 0.15]]

        elif state == self.rootTLLGLLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGRLGL, 0.85], [None, self.rootTLLGRLGR, 0.15]]

        elif state == self.rootTLLGRLGL:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGLLGL, 0.85], [None, self.rootTLLGLLGR, 0.15]]

        elif state == self.rootTLLGRLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[None, self.rootTLOL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTLOR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTLLGRLGL, 0.85], [None, self.rootTLLGRLGR, 0.15]]

        # Tiger on right
        elif state == self.rootTRLGLLGL:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGLLGL, 0.15], [None, self.rootTRLGLLGR, 0.85]]

        elif state == self.rootTRLGLLGR:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGRLGL, 0.15], [None, self.rootTRLGRLGR, 0.85]]

        elif state == self.rootTRLGRLGL:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGLLGL, 0.15], [None, self.rootTRLGLLGR, 0.85]]

        elif state == self.rootTRLGRLGR:
            if action == self.OPENLEFT:
                return [[None, self.rootTROL, 1.0]]
            elif action == self.OPENRIGHT:
                return [[None, self.rootTROR, 1.0]]
            elif action == self.LISTEN:
                return [[None, self.rootTRLGRLGL, 0.15], [None, self.rootTRLGRLGR, 0.85]]

        # No other states have actions
        else:

            return []#[[None, "exit", 1.0]]






# CHANCE = "CHANCE"
#
#
# # Base class for a game state
# class TigerGameBase:
#
#     def __init__(self, parent, next_player, actions):
#         self.parent = parent            # parent node
#         self.next_player = next_player  # which player (0 or 1) has next action
#         self.actions = actions          # available actions at this node
#         self.reach_prob = 1.0
#         #self.reach_prob =
#         self.stateID = None
#         self.payoff = 0
#
#     # Simulate playing an action
#     # Returns the child node of the given action
#     def play(self, action):
#         return self.children[action]
#
#     def is_chance(self):
#         return self.next_player == CHANCE   # Check if chance node
#
#     #@property
#     def inf_set(self):
#         raise NotImplementedError("Please implement information_set method")
#
#     def real_inf_set(self):
#         raise NotImplementedError("Please implement information_set method")
#
#     def stateID(self):
#         return self.stateID
#
#     #def getNextStatesAndProbs(self, state, action):
#
#
#
#
#
#
#
# class TigerGameRoot(TigerGameBase):
#
#     def __init__(self, actions):
#         super().__init__(parent = None, next_player = CHANCE, actions = actions)
#
#         self.actions_history = "."
#         # totState.append(self)
#         # totInfoStates.append(self)
#
#         self.children = {}
#
#         self.childrenProb = {}
#
#         self._chance_prob = 0.5
#
#         self.rewardVector = []
#
#     # Always false, only chance node is at root
#     def is_terminal(self):
#         return False
#
#     def inf_set(self):
#         return "."   #history string (infoset), empty at root
#
#     def real_inf_set(self):
#         return None   #history string (infoset), empty at root
#
#     def chance_prob(self):
#         return self._chance_prob
#
#
#
#
# # These make up entire tree, except for root which is CHANCE
# class TigerGameActionState(TigerGameBase):
#
#     def __init__(self, parent, next_player, actions_history, actions):
#         super().__init__(parent = parent, next_player = next_player, actions = actions)
#
#         self.actions_history = actions_history  # string of history up to this node
#         #self.cards = cards                      # cards players have
#
#
#
#         #public_card = self.cards[0] if self.next_player == player0 else self.cards[1]
#         self._information_set = ".{0}.".format(".".join(self.actions_history))
#         #print(self._information_set)
#         # if self._information_set not in infosets and self.is_terminal() == False:
#         #     infosets.append(self._information_set)
#         #     totInfoStates.append(self)
#
#         self.children = {}
#         self.payoff = 0.0
#
#     # legal actions depending on what actions have been taken
#     def __get_actions_in_next_round(self, a):
#
#         return []
#
#     def inf_set(self):
#         return self._information_set
#
#     def is_terminal(self):
#         return self.actions == []
#
#     def evaluation(self):
#         return self.payoff
#
#
# class TigerGameTerminalState(TigerGameBase):
#
#     def __init__(self, parent, next_player, actions_history, actions):
#         super().__init__(parent = parent, next_player = next_player, actions = actions)
#
#         self.actions_history = actions_history  # string of history up to this node
#
#         self.payoff = 0.0
#
#         #public_card = self.cards[0] if self.next_player == player0 else self.cards[1]
#         self._information_set = ".{0}.".format(".".join(self.actions_history))
#
#         # if self._information_set not in infosets and self.is_terminal() == False:
#         #     infosets.append(self._information_set)
#         #     totInfoStates.append(self)
#
#         self.children = {}
#
#     # legal actions depending on what actions have been taken
#     def __get_actions_in_next_round(self, a):
#
#         return []
#
#     def inf_set(self):
#         return self._information_set
#
#     def is_terminal(self):
#         return self.actions == []
#
#     def evaluation(self):
#         return self.payoff
#
#
#
#
# # Observation chance node
# # These make up entire tree, except for root which is CHANCE
# class TigerGameObsState(TigerGameBase):
#
#     def __init__(self, parent, next_player, actions_history, actions):
#         super().__init__(parent = parent, next_player = CHANCE, actions = actions)
#
#         self.actions_history = actions_history  # string of history up to this node
#         #self.cards = cards                      # cards players have
#
#         self.childrenProb = {}
#
#         #public_card = self.cards[0] if self.next_player == player0 else self.cards[1]
#         self._information_set = ".{0}.".format(".".join(self.actions_history))
#         #print(self._information_set)
#         # if self._information_set not in infosets and self.is_terminal() == False:
#         #     infosets.append(self._information_set)
#         #     totInfoStates.append(self)
#
#         self.children = {}
#         self.payoff = 0.0
#
#     # legal actions depending on what actions have been taken
#     def __get_actions_in_next_round(self, a):
#
#         return []
#
#     def inf_set(self):
#         return self._information_set
#
#     def is_terminal(self):
#         return False#self.actions == []
#
#     def evaluation(self):
#         return self.payoff















