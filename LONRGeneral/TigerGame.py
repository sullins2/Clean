from LONRGeneral.LONR import *
import copy
import numpy as np



# Make two totally separate MDPs for each side.
# Have them share Q Values but not rewards()


class TigerGame(MDP):

    def __init__(self, startState="START", TLProb=0.5, version=1):
        super().__init__(startState=startState)

        # One player MG
        self.N = 1

        # Start with Tiger on Left
        self.version = 1

        # Actions
        self.OPENLEFT = "OL"
        self.LISTEN = "L"
        self.OPENRIGHT = "OR"

        self.TL = "TL"
        self.TR = "TR"

        self.TLProb = TLProb  # Probability that tiger is on left

        # total action list
        self.totalActions = [self.OPENLEFT, self.LISTEN, self.OPENRIGHT]

        self.START = "START"
        self.LEFT = "LEFT"
        self.RIGHT = "RIGHT"
        self.LEFTLEFT = "LEFTLEFT"
        self.LEFTRIGHT = "LEFTRIGHT"
        self.RIGHTLEFT = "RIGHTLEFT"
        self.RIGHTRIGHT = "RIGHTRIGHT"

        self.totalStates = [self.START,
            self.LEFT,self.RIGHT,
            self.LEFTLEFT, self.LEFTRIGHT, self.RIGHTLEFT, self.RIGHTRIGHT]



        self.Q = {}
        self.Q_bu = {}
        self.QSums = {}
        self.QTouched = {}
        self.pi = {}
        self.pi_sums = {}
        self.regret_sums = {}
        self.weights = {}
        for n in range(self.N):
            self.Q[n] = {}
            self.Q_bu[n] = {}
            self.QSums[n] = {}
            self.QTouched[n] = {}
            self.pi[n] = {}
            self.pi_sums[n] = {}
            self.regret_sums[n] = {}
            self.weights[n] = {}
            for s in self.getStates():
                self.Q[n][s] = {}
                self.Q_bu[n][s] = {}
                self.QSums[n][s] = {}
                self.QTouched[n][s] = {}
                # stateInfoSet = self.stateIDtoIS(s)
                # if stateInfoSet == None: continue
                # if stateInfoSet not in self.pi:
                self.pi[n][s] = {}
                self.pi_sums[n][s] = {}
                self.regret_sums[n][s] = {}
                self.weights[n][s] = {}

                for a in self.getActions(s, n):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.QTouched[n][s][a] = 0.0
                    self.pi[n][s][a] = 1.0 / len(self.getActions(s, 0))
                    self.pi_sums[n][s][a] = 0.0
                    self.regret_sums[n][s][a] = 0.0

                    self.weights[n][s][a] = 1.0


    def getStateRep(self, s):

        return s

    def stateIDtoIS(self, state):
        """Returns the information set of state

        """
        return state

    def getActions(self, state, n):
        return [self.OPENLEFT, self.LISTEN, self.OPENRIGHT]

    def getStates(self, init=False):
        return self.totalStates

    def getReward(self, s, a_current, n, a_notN):

        state = s
        action = a_current
        if self.version == 1:

            # If at start
            if state == self.START:

                if action == self.OPENLEFT:
                    return -100.0#[[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0#[[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0#[[self.LEFT, 0.85, -1.0], [self.RIGHT, 0.15, -1.0]]


            # Tiger on left, Listen, GL
            elif state == self.LEFT:
                if action == self.OPENLEFT:
                    return -100.0#[[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0 #[[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0 # [[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]

            # Tiger on left, LISTEN, GR
            elif state == self.RIGHT:
                if action == self.OPENLEFT:
                    return -100.0# [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0 #[[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0#[[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]



            ## Bottom depth
            # Tiger on left
            elif state == self.LEFTLEFT:
                if action == self.OPENLEFT:
                    return -100.0#[[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0#[[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0#[[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]

            elif state == self.LEFTRIGHT:
                if action == self.OPENLEFT:
                    return -100.0 #[[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0# [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]


            elif state == self.RIGHTLEFT:
                if action == self.OPENLEFT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0 #[[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]

            elif state == self.RIGHTRIGHT:
                if action == self.OPENLEFT:
                    # print("state: ", state, " action: ", action)
                    return -100.0 #[[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return 10.0 #[[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]

##################################
        # Top root of Tiger on right
        if self.version == 2:

            # If at start
            if state == self.START:

                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.LEFT, 0.15, -1.0], [self.RIGHT, 0.85, -1.0]]


            # Tiger on right, Listen, GL
            elif state == self.LEFT:
                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.LEFTLEFT, 0.15, -1.0], [self.LEFTRIGHT, 0.85, -1.0]]

            # Tiger on right, LISTEN, GR
            elif state == self.RIGHT:
                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.RIGHTLEFT, 0.15, -1.0], [self.RIGHTRIGHT, 0.85, -1.0]]



            ## Bottom depth
            # Tiger on right
            elif state == self.LEFTLEFT:
                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.LEFTLEFT, 0.15, -1.0], [self.LEFTRIGHT, 0.85, -1.0]]

            elif state == self.LEFTRIGHT:
                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.RIGHTLEFT, 0.15, -1.0], [self.RIGHTRIGHT, 0.85, -1.0]]

            elif state == self.RIGHTLEFT:
                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.LEFTLEFT, 0.15, -1.0], [self.LEFTRIGHT, 0.85, -1.0]]

            elif state == self.RIGHTRIGHT:
                if action == self.OPENLEFT:
                    return 10.0 ##[[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return -100.0 ##[[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return -1.0 ##[[self.RIGHTLEFT, 0.15, -1.0], [self.RIGHTRIGHT, 0.85, -1.0]]


        ##return None

        # return 0.0
        # if self.version == 1:
        #     if a_current == self.OPENLEFT:
        #         return -100.0
        #     elif a_current == self.OPENRIGHT:
        #         return 10.0
        #     else:
        #         return -1.0
        # if self.version == 2:
        #     if a_current == self.OPENLEFT:
        #         return 10.0
        #     elif a_current == self.OPENRIGHT:
        #         return -100.0
        #     else:
        #         return -1.0
        # print("MISSED REWARD")

        # return 0.0
        # if a_current == self.LISTEN:
        #     return -1.0
        # else:
        #     return 0.0

        # if s == self.rootTLOL:
        #     return -100.0
        # # Tiger left, open RIGHT
        # elif s == self.rootTLOR:
        #     return 10.0
        #
        # elif s == self.rootTL:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        #
        # elif s == self.rootTLLGL or s == self.rootTLLGR:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        # elif s == self.rootTLLGLLGL or s == self.rootTLLGLLGR or s == self.rootTLLGRLGL or s == self.rootTLLGRLGR:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        #
        #
        # else:
        #     print("Not caught: ", s)
        #     return 0.0

    def isTerminal(self, state):
        return False
        #if state == self.rootTLOL or state == self.rootTLOR:
        #    return True
        #else:
        #   return False

    def getNextStatesAndProbs(self, state, action, n_current):
        # Top root of Tiger on left
        if self.version == 1:

            # If at start
            if state == self.START:

                if action == self.OPENLEFT:
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.LEFT, 0.85, -1.0], [self.RIGHT, 0.15, -1.0]]


            # Tiger on left, Listen, GL
            elif state == self.LEFT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]

            # Tiger on left, LISTEN, GR
            elif state == self.RIGHT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]



            ## Bottom depth
            # Tiger on left
            elif state == self.LEFTLEFT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]

            elif state == self.LEFTRIGHT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]

            elif state == self.RIGHTLEFT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]

            elif state == self.RIGHTRIGHT:
                if action == self.OPENLEFT:
                    # print("state: ", state, " action: ", action)
                    return [[None, 1.0, -100.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, 10.0]]
                elif action == self.LISTEN:
                    return [[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]

        # Top root of Tiger on right
        if self.version == 2:

            # If at start
            if state == self.START:

                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.LEFT, 0.15, -1.0], [self.RIGHT, 0.85, -1.0]]


            # Tiger on right, Listen, GL
            elif state == self.LEFT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.LEFTLEFT, 0.15, -1.0], [self.LEFTRIGHT, 0.85, -1.0]]

            # Tiger on right, LISTEN, GR
            elif state == self.RIGHT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.RIGHTLEFT, 0.15, -1.0], [self.RIGHTRIGHT, 0.85, -1.0]]



            ## Bottom depth
            # Tiger on right
            elif state == self.LEFTLEFT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.LEFTLEFT, 0.15, -1.0], [self.LEFTRIGHT, 0.85, -1.0]]

            elif state == self.LEFTRIGHT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.RIGHTLEFT, 0.15, -1.0], [self.RIGHTRIGHT, 0.85, -1.0]]

            elif state == self.RIGHTLEFT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.LEFTLEFT, 0.15, -1.0], [self.LEFTRIGHT, 0.85, -1.0]]


            # elif state == self.RIGHTLEFT:
            #     if action == self.OPENLEFT:
            #         return [[None, 1.0, -100.0]]
            #     elif action == self.OPENRIGHT:
            #         return [[None, 1.0, 10.0]]
            #     elif action == self.LISTEN:
            #         return [[self.LEFTLEFT, 0.85, -1.0], [self.LEFTRIGHT, 0.15, -1.0]]


            elif state == self.RIGHTRIGHT:
                if action == self.OPENLEFT:
                    return [[None, 1.0, 10.0]]
                elif action == self.OPENRIGHT:
                    return [[None, 1.0, -100.0]]
                elif action == self.LISTEN:
                    return [[self.RIGHTLEFT, 0.15, -1.0], [self.RIGHTRIGHT, 0.85, -1.0]]

            # elif state == self.RIGHTRIGHT:
            #     if action == self.OPENLEFT:
            #         # print("state: ", state, " action: ", action)
            #         return [[None, 1.0, -100.0]]
            #     elif action == self.OPENRIGHT:
            #         return [[None, 1.0, 10.0]]
            #     elif action == self.LISTEN:
            #         return [[self.RIGHTLEFT, 0.85, -1.0], [self.RIGHTRIGHT, 0.15, -1.0]]


        # No other states have actions
        else:
            print("None: ", state, "  ", action)
            return []

    def getMove(self, s, a):
        if self.isTerminal(s): return None
        # print("S: ", s , "  a: ", a )
        nextPossStates = self.getNextStatesAndProbs(s, a, 0)
        if nextPossStates == None: return None
        # print("NPS: ", nextPossStates)
        if len(nextPossStates) == 1:
            currentState = nextPossStates[0][0]
        else:
            nst = []
            nspt = []
            for nstate, nsp, nr in nextPossStates:
                nst.append(nstate)
                nspt.append(nsp)
            currentState = np.random.choice(nst, p=nspt)
        return currentState




























class TigerGame2(MDP):

    # version: 0 is for LONR-A
    #          1 is for left mdp
    #          2 is for right mdp
    def __init__(self, startState=None, TLProb=0.5, version=0):
        super().__init__(startState=startState)

        # One player MG
        self.N = 1

        self.version = 1

        # Actions
        self.OPENLEFT = "OL"
        self.LISTEN = "L"
        self.OPENRIGHT = "OR"

        self.TL = "TL"
        self.TR = "TR"

        self.TLProb = TLProb  # Probability that tiger is on left

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
        self.rootTLLGLOL = "rootTLLGLOL"
        self.rootTLLGLOR = "rootTLLGLOR"

        self.rootTLLGR = "rootTLLGR"     # Tiger on LEFT, LISTEN, GR
        self.rootTLLGROL = "rootTLLGROL"
        self.rootTLLGROR = "rootTLLGROR"

        self.rootTRLGL = "rootTRLGL"     # Tiger on RIGHT, LISTEN, GL
        self.rootTRLGLOL = "rootTRLGLOL"
        self.rootTRLGLOR = "rootTRLGLOR"

        self.rootTRLGR = "rootTRLGR"     # Tiger on RIGHT, LISTEN, GR
        self.rootTRLGROL = "rootTRLGROL"
        self.rootTRLGROR = "rootTRLGROR"

        # depth = 4
        self.rootTLLGLLGL = "rootTLLGLLGL" # Tiger on left, Listen, Growl Left, Listen, Growl Left
        self.rootTLLGLLGLOL = "rootTLLGLLGLOL"
        self.rootTLLGLLGLOR = "rootTLLGLLGLOR"

        self.rootTLLGLLGR = "rootTLLGLLGR"
        self.rootTLLGLLGROL = "rootTLLGLLGROL"
        self.rootTLLGLLGROR = "rootTLLGLLGROR"

        self.rootTLLGRLGL = "rootTLLGRLGL"
        self.rootTLLGRLGLOL = "rootTLLGRLGLOL"
        self.rootTLLGRLGLOR = "rootTLLGRLGLOR"

        self.rootTLLGRLGR = "rootTLLGRLGR"
        self.rootTLLGRLGROL = "rootTLLGRLGROL"
        self.rootTLLGRLGROR = "rootTLLGRLGROR"

        self.rootTRLGLLGL = "rootTRLGLLGL" # Tiger on right,
        self.rootTRLGLLGLOL = "rootTRLGLLGLOL"
        self.rootTRLGLLGLOR = "rootTRLGLLGLOR"

        self.rootTRLGLLGR = "rootTRLGLLGR"
        self.rootTRLGLLGROL = "rootTRLGLLGROL"
        self.rootTRLGLLGROR = "rootTRLGLLGROR"

        self.rootTRLGRLGL = "rootTRLGRLGL"
        self.rootTRLGRLGLOL = "rootTRLGRLGLOL"
        self.rootTRLGRLGLOR = "rootTRLGRLGLOR"

        self.rootTRLGRLGR = "rootTRLGRLGR"
        self.rootTRLGRLGROL = "rootTRLGRLGROL"
        self.rootTRLGRLGROR = "rootTRLGRLGROR"

        self.totalStates = [
                            self.rootTL, self.rootTR, self.rootTLOL, self.rootTLOR, self.rootTROL, self.rootTROR,

                            self.rootTLLGL, self.rootTLLGR, self.rootTRLGL, self.rootTRLGR,
                            self.rootTLLGLOL, self.rootTLLGROL, self.rootTRLGLOL, self.rootTRLGROL,
                            self.rootTLLGLOR, self.rootTLLGROR, self.rootTRLGLOR, self.rootTRLGROR,

                            self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR,
                            self.rootTLLGLLGLOL, self.rootTLLGLLGROL, self.rootTLLGRLGLOL, self.rootTLLGRLGROL,
                            self.rootTLLGLLGLOR, self.rootTLLGLLGROR, self.rootTLLGRLGLOR, self.rootTLLGRLGROR,

                            self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR,
                            self.rootTRLGLLGLOL, self.rootTRLGLLGROL, self.rootTRLGRLGLOL, self.rootTRLGRLGROL,
                            self.rootTRLGLLGLOR, self.rootTRLGLLGROR, self.rootTRLGRLGLOR, self.rootTRLGRLGROR]

        self.totalStatesLeft = [self.rootTL, self.rootTLOL, self.rootTLOR,
                                self.rootTLLGL, self.rootTLLGR,
                                self.rootTLLGLOL, self.rootTLLGROL,
                                self.rootTLLGLOR, self.rootTLLGROR,
                            self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR,
                            self.rootTLLGLLGLOL, self.rootTLLGLLGROL, self.rootTLLGRLGLOL, self.rootTLLGRLGROL,
                            self.rootTLLGLLGLOR, self.rootTLLGLLGROR, self.rootTLLGRLGLOR, self.rootTLLGRLGROR
                             ]

        self.totalStatesRight = [self.rootTR, self.rootTROL, self.rootTROR,
                            self.rootTRLGL, self.rootTRLGR,
                            self.rootTRLGLOL, self.rootTRLGROL,
                            self.rootTRLGLOR, self.rootTRLGROR,

                            self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR,
                            self.rootTRLGLLGLOL, self.rootTRLGLLGROL, self.rootTRLGRLGLOL, self.rootTRLGRLGROL,
                            self.rootTRLGLLGLOR, self.rootTRLGLLGROR, self.rootTRLGRLGLOR, self.rootTRLGRLGROR]

        # self.totalStates = [self.root,
        #                     self.rootTL, self.rootTR,
        #                     self.rootTLOL, self.rootTLOR, self.rootTROL, self.rootTROR,
        #                     self.rootTLLGL, self.rootTLLGR, self.rootTRLGL, self.rootTRLGR,
        #                     self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR,
        #                     self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR]
        #
        # self.totalStatesLeft = [self.rootTL,
        #                         self.rootTLLGL, self.rootTLLGR,
        #                         self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR,
        #                         self.rootTLOL, self.rootTLOR]
        #
        # self.totalStatesRight = [self.rootTR,
        #                          self.rootTRLGL, self.rootTRLGR,
        #                          self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR,
        #                          self.rootTROL, self.rootTROR]



        self.Q = {}
        self.Q_bu = {}
        self.QSums = {}
        self.pi = {}
        self.pi_sums = {}
        self.regret_sums = {}
        for n in range(self.N):
            self.Q[n] = {}
            self.Q_bu[n] = {}
            self.QSums[n] = {}
            self.pi[n] = {}
            self.pi_sums[n] = {}
            self.regret_sums[n] = {}
            for s in self.getStates(init=True):
                self.Q[n][s] = {}
                self.Q_bu[n][s] = {}
                self.QSums[n][s] = {}

                stateInfoSet = self.stateIDtoIS(s)
                if stateInfoSet == None: continue
                if stateInfoSet not in self.pi:
                    self.pi[n][stateInfoSet] = {}
                    self.pi_sums[n][stateInfoSet] = {}
                    self.regret_sums[n][stateInfoSet] = {}



                # self.pi[n][s] = {}
                # self.regret_sums[n][s] = {}
                # self.pi_sums[n][s] = {}
                for a in self.getActions(s, n):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0

                    if stateInfoSet == None: continue
                    if stateInfoSet not in self.pi:
                        self.pi[n][stateInfoSet][a] = 1.0 / len(self.getActions(stateInfoSet, 0))
                        self.pi_sums[n][stateInfoSet][a] = 0.0
                        self.regret_sums[n][stateInfoSet][a] = 0.0


    def getStateRep(self, s):

        return self.stateIDtoIS(s)

    def stateIDtoIS(self, state):
        """Returns the information set of state

        """

        # uses rootTLxxx for pi, regret, etc
        # Ex:  rootTL is infoset for both first action nodes on left and right
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

        else:
            return state

        # elif state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
        #     return state
        # elif state == self.root:
        #     return state
        # else:
        #     return None

    def getActions(self, state, n):
        if state == self.root:
            return [self.TL, self.TR]
        # elif state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
        #     return [] #["exit"]
        else:
            return [self.OPENLEFT, self.LISTEN, self.OPENRIGHT]

    def getStates(self, init=False):

        if init:
            return self.totalStates
        else:
            if self.version == 1:
                return self.totalStatesLeft
            else:
                return self.totalStatesRight

        # if self.version == 1:
        #     return self.totalStatesLeft
        # elif self.version == 2:
        #     return self.totalStatesRight
        # else:
        #     return self.totalStates

    #self, s, a_current, n, a_notN
    def getReward(self, s, a_current, n, a_notN):

        # if s == self.root:
        #     return 0.0

        if s == self.rootTLOL or s == self.rootTLLGLOL or s ==self.rootTLLGROL or s == self.rootTLLGLLGLOL or s == self.rootTLLGLLGROL or s == self.rootTLLGRLGLOL or s == self.rootTLLGRLGROL:
            return -100.0
        if s == self.rootTLOR or s == self.rootTLLGLOR or s == self.rootTLLGROR or s == self.rootTLLGLLGLOR or s == self.rootTLLGLLGROR or s == self.rootTLLGRLGLOR or s == self.rootTLLGRLGROR:
            return 10.0

        if s == self.rootTROL or s == self.rootTRLGLOL or s == self.rootTRLGROL or s == self.rootTRLGLLGLOL or s == self.rootTRLGLLGROL or s == self.rootTRLGRLGLOL or s == self.rootTRLGRLGROL:
            return 10.0
        if s == self.rootTROR or s == self.rootTRLGLOR or s == self.rootTRLGROR or s == self.rootTRLGLLGLOR or s == self.rootTRLGLLGROR or s == self.rootTRLGRLGLOR or s == self.rootTRLGRLGROR:
            return -100.0
        # if s == self.rootTLOL:
        #     return -100.0
        # # Tiger left, open RIGHT
        # elif s == self.rootTLOR:
        #     return 10.0
        # # Tiger right, open LEFT
        # elif s == self.rootTROL:
        #     return 10.0
        # # Tiger right, open RIGHT
        # elif s == self.rootTROR:
        #     return -100.0

        if a_current == self.LISTEN:
            return -1.0
        else:
            return 0.0

        # elif s == self.rootTL or s == self.rootTR:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        #
        # elif s == self.rootTLLGL or s == self.rootTLLGR or s == self.rootTRLGL or s == self.rootTRLGR:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        # elif s == self.rootTLLGLLGL or s == self.rootTLLGLLGR or s == self.rootTLLGRLGL or s == self.rootTLLGRLGR:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        # elif s == self.rootTRLGLLGL or s == self.rootTRLGLLGR or s == self.rootTRLGRLGL or s == self.rootTRLGRLGR:
        #     if a_current == self.LISTEN:
        #         return -1.0
        #     else:
        #         return 0.0
        #
        #
        # else:
        #     print("Not caught: ", s)
        #     return 0.0

    def isTerminal(self, state):

        stateStr = state
        length = len(state)
        if state[-2] == "O":
            #print("TERMINAL: ", state)
            return True

        else:
            #print("NOT TERMINAL: ", state)
            return False
        # if state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
        #     return True
        # else:
        #     return False


    def getNextStatesAndProbs(self, state, action, n_current):

        #self.UP, upState, 1.0 - self.noise]

        print("state: ", state, " action: ", action)
        # Top most root
        if state == self.root:
            if action == None:
                return [[self.rootTL, self.TLProb, 0.0], [self.rootTR, 1.0 - self.TLProb, 0.0]]

            # Shouldn't be reached
            # elif action == "TL":
            #     return [[self.rootTL, 0.5, 0.0]]
            # else:
            #     return [[self.rootTR, 0.5, 0.0]]

        # elif state == self.rootTLOL:
        #     return []

        # Top root of Tiger on left
        elif state == self.rootTL:

            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGL, 0.85, -1.0], [self.rootTLLGR, 0.15, -1.0]]

        # Top root of Tiger on right
        elif state == self.rootTR:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGL, 0.15, -1.0], [self.rootTRLGR, 0.85, -1.0]]

        # Tiger on left, Listen, GL
        elif state == self.rootTLLGL:
            if action == self.OPENLEFT:
                return [[self.rootTLLGLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLLGLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGLLGL, 0.85, -1.0], [self.rootTLLGLLGR, 0.15, -1.0]]

        # Tiger on left, LISTEN, GR
        elif state == self.rootTLLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[self.rootTLLGROL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLLGROR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGRLGL, 0.85, -1.0], [self.rootTLLGRLGR, 0.15, -1.0]]

        # Tiger on RIGHT, Listen, GL
        elif state == self.rootTRLGL:
            if action == self.OPENLEFT:
                return [[self.rootTRLGLOL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTRLGLOR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGLLGL, 0.15, -1.0], [self.rootTRLGLLGR, 0.85, -1.0]]

        # Tiger on RIGHT, LISTEN, GR
        elif state == self.rootTRLGR:
            if action == self.OPENLEFT:
                return [[self.rootTRLGROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTRLGROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGRLGL, 0.15, -1.0], [self.rootTRLGRLGR, 0.85, -1.0]]

    ## Bottom depth
        # Tiger on left
        elif state == self.rootTLLGLLGL:
            if action == self.OPENLEFT:
                return [[self.rootTLLGLLGLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLLGLLGLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGLLGL, 0.85, -1.0], [self.rootTLLGLLGR, 0.15, -1.0]]

        elif state == self.rootTLLGLLGR:
            if action == self.OPENLEFT:
                return [[self.rootTLLGLLGROL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLLGLLGROR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGRLGL, 0.85, -1.0], [self.rootTLLGRLGR, 0.15, -1.0]]

        elif state == self.rootTLLGRLGL:
            if action == self.OPENLEFT:
                return [[self.rootTLLGRLGLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLLGRLGLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGLLGL, 0.85, -1.0], [self.rootTLLGLLGR, 0.15, -1.0]]

        elif state == self.rootTLLGRLGR:
            if action == self.OPENLEFT:
                return [[self.rootTLLGRLGROL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLLGRLGROR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGRLGL, 0.85, -1.0], [self.rootTLLGRLGR, 0.15, -1.0]]

        # Tiger on right
        elif state == self.rootTRLGLLGL:
            if action == self.OPENLEFT:
                return [[self.rootTRLGLLGLOL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTRLGLLGLOR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGLLGL, 0.15, -1.0], [self.rootTRLGLLGR, 0.85, -1.0]]

        elif state == self.rootTRLGLLGR:
            if action == self.OPENLEFT:
                return [[self.rootTRLGLLGROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTRLGLLGROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGRLGL, 0.15, -1.0], [self.rootTRLGRLGR, 0.85, -1.0]]

        elif state == self.rootTRLGRLGL:
            if action == self.OPENLEFT:
                return [[self.rootTRLGRLGLOL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTRLGRLGLOR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGLLGL, 0.15, -1.0], [self.rootTRLGLLGR, 0.85, -1.0]]

        elif state == self.rootTRLGRLGR:
            if action == self.OPENLEFT:
                return [[self.rootTRLGRLGROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTRLGRLGROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGRLGL, 0.15, -1.0], [self.rootTRLGRLGR, 0.85, -1.0]]

        # No other states have actions
        else:
            print("Missed: s: ", state, " a: ", action)
            return []


    def getMove(self, s, a):
        if self.isTerminal(s): return None
        #print("S: ", s , "  a: ", a )
        nextPossStates = self.getNextStatesAndProbs(s, a, 0)
        if nextPossStates == None: return None
        #print("NPS: ", nextPossStates)
        if len(nextPossStates) == 1:
            currentState = nextPossStates[0][0]
        else:
            nst = []
            nspt = []
            for nstate, nsp, nr in nextPossStates:
                nst.append(nstate)
                nspt.append(nsp)
            currentState = np.random.choice(nst, p=nspt)
        return currentState


# TODO: Make each MDP seperate and feed both into LONR
class TG(TigerGame):


    def __init__(self, startState=None, TLProb=0.5, version=None):
        super().__init__(startState=startState, version=version)
