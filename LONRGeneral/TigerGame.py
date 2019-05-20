from LONRGeneral.LONR import *
import copy
import numpy as np



class TigerGame(MDP):

    def __init__(self, startState=None, TLProb=0.5):
        super().__init__(startState=startState)

        # Two player MG
        self.N = 1

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
                                self.rootTLLGL, self.rootTLLGR,
                                self.rootTLLGLLGL, self.rootTLLGLLGR, self.rootTLLGRLGL, self.rootTLLGRLGR,
                                self.rootTLOL, self.rootTLOR,
                                self.rootTROL, self.rootTROR]

        self.totalStatesRight = [self.rootTR,
                                 self.rootTRLGL, self.rootTRLGR,
                                 self.rootTRLGLLGL, self.rootTRLGLLGR, self.rootTRLGRLGL, self.rootTRLGRLGR,
                                 self.rootTROL, self.rootTROR,
                                 self.rootTLOL, self.rootTLOR]


        n = 0
        # for s in self.getStates():
        #
        #     for a in self.getActions(s, n):
        #
        #         # Add entry into Qs
        #         if s not in self.Q:
        #             self.Q[s] = {}
        #             self.Q_bu[s] = {}
        #             self.Qsums[s] = {}
        #         self.Q[s][a] = 0.0
        #         self.Q_bu[s][a] = 0.0
        #         self.Qsums[s][a] = 0.0
        #
        #         # Add all TLxxx as info sets
        #         stateInfoSet = self.tigergame.stateIDtoIS(s)
        #         if stateInfoSet == None: continue
        #         if stateInfoSet not in self.pi:
        #             self.pi[stateInfoSet] = {}
        #             self.pi_sums[stateInfoSet] = {}
        #             self.regret_sums[stateInfoSet] = {}
        #
        #         self.pi[stateInfoSet][a] = 1.0 / len(self.tigergame.getActions(stateInfoSet))
        #         self.pi_sums[stateInfoSet][a] = 0.0
        #         self.regret_sums[stateInfoSet][a] = 0.0

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
            for s in self.getStates():
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


                    # self.pi[n][s][a] = 1.0 / len(self.getActions(s, n))
                    # self.regret_sums[n][s][a] = 0.0
                    # self.pi_sums[n][s][a] = 0.0


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



        elif state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
            return state
        elif state == self.root:
            return state
        else:
            return None

    def getActions(self, state, n):
        if state == self.root:
            return [self.TL, self.TR]
        elif state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
            return ["exit"]
        else:
            return [self.OPENLEFT, self.LISTEN, self.OPENRIGHT]

    def getStates(self):
        return self.totalStates

    #self, s, a_current, n, a_notN
    def getReward(self, s, a_current, n, a_notN):


        # return None
        # Tiger left, open LEFT
        if s == self.root:
            return 0.0
        elif s == self.rootTL or s == self.rootTR:
            if a_current == self.LISTEN:
                return -1.0
            else:
                return 0.0
        if s == self.rootTLOL:
            return -100.0
        # Tiger left, open RIGHT
        elif s == self.rootTLOR:
            return 10.0
        # Tiger right, open LEFT
        elif s == self.rootTROL:
            return 10.0
        # Tiger right, open RIGHT
        elif s == self.rootTROR:
            return -100.0

        elif s == self.rootTLLGL or s == self.rootTLLGR or s == self.rootTRLGL or s == self.rootTRLGR:
            if a_current == self.LISTEN:
                return -1.0
            else:
                return 0.0
        elif s == self.rootTLLGLLGL or s == self.rootTLLGLLGR or s == self.rootTLLGRLGL or s == self.rootTLLGRLGR:
            if a_current == self.LISTEN:
                return -1.0
            else:
                return 0.0
        elif s == self.rootTRLGLLGL or s == self.rootTRLGLLGR or s == self.rootTRLGRLGL or s == self.rootTRLGRLGR:
            if a_current == self.LISTEN:
                return -1.0
            else:
                return 0.0


        else:
            print("Not caught: ", s)
            return 0.0

    def isTerminal(self, state):
        if state == self.rootTLOL or state == self.rootTLOR or state == self.rootTROL or state == self.rootTROR:
            return True
        else:
            return False


    def getNextStatesAndProbs(self, state, action, n_current):

        #self.UP, upState, 1.0 - self.noise]

        # Top most root
        if state == self.root:
            if action == None:
                return [[self.rootTL, self.TLProb, 0.0], [self.rootTR, 1.0 - self.TLProb, 0.0]]

            # Shouldn't be reached
            elif action == "TL":
                return [[self.rootTL, 0.5, 0.0]]
            else:
                return [[self.rootTR, 0.5, 0.0]]

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
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGLLGL, 0.85, -1.0], [self.rootTLLGLLGR, 0.15, -1.0]]

        # Tiger on left, LISTEN, GR
        elif state == self.rootTLLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGRLGL, 0.85, -1.0], [self.rootTLLGRLGR, 0.15, -1.0]]

        # Tiger on RIGHT, Listen, GL
        elif state == self.rootTRLGL:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGLLGL, 0.15, -1.0], [self.rootTRLGLLGR, 0.85, -1.0]]

        # Tiger on RIGHT, LISTEN, GR
        elif state == self.rootTRLGR:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGRLGL, 0.15, -1.0], [self.rootTRLGRLGR, 0.85, -1.0]]

    ## Bottom depth
        # Tiger on left
        elif state == self.rootTLLGLLGL:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGLLGL, 0.85, -1.0], [self.rootTLLGLLGR, 0.15, -1.0]]

        elif state == self.rootTLLGLLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGRLGL, 0.85, -1.0], [self.rootTLLGRLGR, 0.15, -1.0]]

        elif state == self.rootTLLGRLGL:
            if action == self.OPENLEFT:
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGLLGL, 0.85, -1.0], [self.rootTLLGLLGR, 0.15, -1.0]]

        elif state == self.rootTLLGRLGR:
            if action == self.OPENLEFT:
                #print("state: ", state, " action: ", action)
                return [[self.rootTLOL, 1.0, -100.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTLOR, 1.0, 10.0]]
            elif action == self.LISTEN:
                return [[self.rootTLLGRLGL, 0.85, -1.0], [self.rootTLLGRLGR, 0.15, -1.0]]

        # Tiger on right
        elif state == self.rootTRLGLLGL:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGLLGL, 0.15, -1.0], [self.rootTRLGLLGR, 0.85, -1.0]]

        elif state == self.rootTRLGLLGR:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGRLGL, 0.15, -1.0], [self.rootTRLGRLGR, 0.85, -1.0]]

        elif state == self.rootTRLGRLGL:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGLLGL, 0.15, -1.0], [self.rootTRLGLLGR, 0.85, -1.0]]

        elif state == self.rootTRLGRLGR:
            if action == self.OPENLEFT:
                return [[self.rootTROL, 1.0, 10.0]]
            elif action == self.OPENRIGHT:
                return [[self.rootTROR, 1.0, -100.0]]
            elif action == self.LISTEN:
                return [[self.rootTRLGRLGL, 0.15, -1.0], [self.rootTRLGRLGR, 0.85, -1.0]]

        # No other states have actions
        else:

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