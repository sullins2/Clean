
import numpy as np
import math

class MDP(object):

    def __init__(self, startState=None):

        # Check if these are used
        self.totalPlayers = 1
        self.totalStates = 1

        # Initialized here but must be defined/created in created class
        self.Q = None
        self.Q_bu = None
        self.QSums = None
        self.pi = None
        self.pi_sums = None
        self.regret_sums = None

        self.weights = None

        self.QSum = [1,2]

        self.startState = startState

    def getActions(self, s, n):
        """ Returns list of actions

        """
        raise NotImplementedError("Please implement getActions method")

    def getNextStatesAndProbs(self, s, a_current, n_current):
        """ Returns list of [ [next_state, prob, rew] ]

        """
        raise NotImplementedError("Please implement getNextStatesAndProbs method")


    def getReward(self, s, a_current, n, a_notN):
        """ s=state, a_current=actionOfPlayer n, n is current player, a_notN is action of other player
            Joint Actions required for Markov Games
            a_notN can be set to 0 for MDPs and ignored

        Returns reward(s, a, a_other)

        """
        raise NotImplementedError("Please implement getReward method")

    def getStates(self):
        """Return List of States

        """
        raise NotImplementedError("Please implement getStates method")

    def isTerminal(self, s):
        raise NotImplementedError("Please implement isTerminal method")


    # The following are only needed for O-LONR

    def getMove(self, s, a):
        """Used to get actual move in the environment for O-LONR
        """
        raise NotImplementedError("Please implement getMove method")

    def getStateRep(self, s):
        """ O-LONR use only

            For MDPs: simply returns state
            For TigerGame: Maps both to one (EX: TLL & TRL -> TLL) aka one set of Q Values
        """
        raise NotImplementedError("Please implement getStateRep method")

class LONR(object):

    # M: Markov game (MDP, markov game, tiger game)

    def __init__(self, M=None, gamma=1.0, alpha=1.0, epsilon=10, alphaDecay=1.0, RMPLUS=False, DCFR=False, VI=True, randomize=False, showSettings=True):




        self.M = M
        #self.N = M.totalPlayers
        #self.S = M.totalStates

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.alphaDecay = alphaDecay

        self.RMPLUS = RMPLUS
        self.DCFR = DCFR

        self.VI = VI

        self.randomize = randomize

        self.showSettings = showSettings

        self.alt = 1

        self.QSums = []
        self.iters = []

        # GRIDWORLD
        self.runningSum = 0.0

        # NoSDE
        self.QSumSEND = []
        self.QSumSendRunningSum = 0.0
        self.QSumKEEP = []
        self.QSumKEEPRunningSum = 0.0
        self.QSumNOOP = []
        self.QSumNOOPRunningSum = 0.0
        self.pisumSEND = []
        self.pisumKEEP = []
        self.pisumNOOP = []

        # SoccerGame A12
        self.QSumSoccerB21Right = []
        self.QSumSoccerB21RightRunningSum = 0.0

        # Tiger Game
        self.L = []
        self.LRunningSum = 0.0
        self.OL = []
        self.OLRunningSum = 0.0
        self.OR = []
        self.ORRunningSum = 0.0

        self.rootTLLGLLGL = []
        self.rootTLLGLLGLRS = 0.0
        self.rootTLLGRLGR = []
        self.rootTLLGRLGRRS = 0.0
        self.rootTLLGLLGR = []
        self.rootTLLGLLGRRS = 0.0
        self.rootTLLGRLGL = []
        self.rootTLLGRLGLRS = 0.0



        self.GRIDWORLD = False
        self.SOCCER = False
        self.NOSDE = False
        self.TIGERGAME = False

        self.EXP3 = True

        if self.showSettings:
            print("LONR initialized with:")
            print("   gamma: ", self.gamma)
            print("   alpha: ", self.alpha)
            print(" epsilon: ", self.epsilon)
            print("     rm+: ", self.RMPLUS)
            print("    DCFR: ", self.DCFR)
            print("      VI: ", self.VI)


    def lonr_value_iteration(self, iterations=-1, log=-1):



        print("Starting training..")
        tt = 0
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha: ", self.alpha)


            self._lonr_value_iteration(t=t)
            #print("")
            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)



            self.iters.append(t)

            if self.GRIDWORLD:
                if self.M.noise > 0:
                    self.runningSum += self.M.Q[0][36][3]
                else:
                    self.runningSum += self.M.Q[0][36][0]
                self.QSums.append(float(self.runningSum) / float(t))
            elif self.NOSDE:
                # Player 0, state left, KEEP
                self.QSumSendRunningSum += self.M.Q[0][1]["SEND"]
                self.QSumSEND.append(float(self.QSumSendRunningSum) / float(t))
                # Player 0, state left
                self.QSumKEEPRunningSum += self.M.Q[0][1]["KEEP"]
                self.QSumKEEP.append(float(self.QSumKEEPRunningSum) / float(t))

                self.QSumNOOPRunningSum += self.M.Q[0][2]["NOOP"]
                self.QSumNOOP.append(float(self.QSumNOOPRunningSum) / float(t))


            elif self.SOCCER:
                #print("VAL: ", self.M.Q[0]["B21"])
                self.QSumSoccerB21RightRunningSum += self.M.Q[0]["B21"][3]
                self.QSumSoccerB21Right.append(float(self.QSumSoccerB21RightRunningSum) / float(t))
            tt = t

        # print("Finish Training")
        # print("LastAVG: ", float(self.runningSum) / float(tt))
        # print("Avg: ", float(self.runningSum) / float(iterations))
        #print(self.QSumSEND)
        #print(self.QSums)
        #print("PRINTED QSUMS")
        if self.GRIDWORLD:
            return self.QSums, self.iters
        elif self.NOSDE:

            return self.QSumSEND, self.QSumKEEP, self.QSumNOOP,  self.iters
        elif self.SOCCER:
            return self.QSumSoccerB21Right, self.iters


    def QUpdate(self, n, s, a_current2, randomS=None):

        #print("IN:  s: ", s, " RANDOMS: ", randomS)
        for a_current in a_current2:

            if self.M.isTerminal(s):

                # Value iteration
                if self.VI == True:
                    # All except tiger game
                    if randomS is None:
                        reward = self.M.getReward(s, a_current, 0, 0)
                        if reward == None:
                            reward = 0.0
                        self.M.Q_bu[n][s][a_current] = reward #(1.0 - self.alpha) * self.M.Q[n][s][a_current] + self.alpha * reward #self.M.getReward(s, a_current, 0, 0)
                    # Tiger game value iteration
                    # else:
                    #     reward = self.M.getReward(randomS, a_current, 0, 0)
                    #     print("HERE: terminal s: ", s)
                    #     if reward == None:
                    #         reward = 0.0
                    #         print("NOOOOO")
                    #     else:
                    #         print("DDDDDDD")
                    #         #self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha) * self.M.Q[n][s][a_current] + self.alpha * reward
                    continue

                # Online
                else:
                    if randomS is None:

                        reward = self.M.getReward(s, a_current, 0,0)
                        if reward == None:
                            reward = 0.0
                        self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] + self.alpha*reward#self.M.getReward(s, a_current, 0,0)
                    else:
                        reward = self.M.getReward(randomS, a_current, 0, 0)
                        if reward == None:
                            reward = 0.0

                        self.M.Q_bu[n][self.M.getStateRep(s)][a_current] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(s)][a_current] + self.alpha * reward#self.M.getReward(randomS, a_current, 0, 0)
                    continue


            # if self.VI == True and randomS is not None:
            #     succs = self.M.getNextStatesAndProbs(s, a_current, n)
            #     #print("s: ", s, " a:", a_current, "   succs: ", succs)
            #     Value = 0.0
            #
            #     # Loop thru each next state and prob
            #     for s_prime, prob, reward in succs:
            #         # print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
            #         tempValue = 0.0
            #
            #         # Loop thru actions in s_prime for player n
            #         for a_current_prime in self.M.getActions(s_prime, n):
            #             tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]
            #
            #         #reww = self.M.getReward(randomS, a_current, n, 0)
            #         reww = self.M.getReward(s, a_current, n, 0)
            #
            #         #print("REWW: ", reww)
            #         if reww is None:
            #             Value += prob * (reward + self.gamma * tempValue)
            #         else:
            #             Value += prob * (reww + self.gamma * tempValue)
            #
            #     self.M.Q_bu[n][self.M.getStateRep(s)][a_current] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(s)][a_current] + (self.alpha) * Value
            #     print("SETTING: s: ", self.M.getStateRep(s), " a_current: ", a_current, " to: ", self.M.Q_bu[n][self.M.getStateRep(s)][a_current])
            # else:
            # print("S: ", s)

            # Get next states and transition probs for MDP
            # Get next states and other players policy for MG
            succs = self.M.getNextStatesAndProbs(s, a_current, n)
            # print("s: ", s, " a:", a_current, "   succs: ", succs)

            #print("  SUCCS: ", succs)
            Value = 0.0

            #Loop thru each next state and prob
            for s_prime, prob, reward in succs:
                #print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
                tempValue = 0.0

                # Loop thru actions in s_prime for player n
                for a_current_prime in self.M.getActions(s_prime, n):
                    tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]


                reww = self.M.getReward(s, a_current, n, 0)


                if reww is None:
                    Value += prob * (reward + self.gamma * tempValue)
                else:
                    Value += prob * (reww + self.gamma * tempValue)

            self.M.Q_bu[n][self.M.getStateRep(s)][a_current] = (1.0 - self.alpha)*self.M.Q[n][self.M.getStateRep(s)][a_current] + (self.alpha)*Value



    def QBackup(self, n, WW=None):

        if WW is None:
            for s in self.M.getStates():
                for a in self.M.getActions(s, n):
                    #self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
                    self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
                    self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]
                    #if s == 36 and a == self.M.UP:
                    #self.M.QSum.append(self.M.Q[n][s][a])
        else:
            for s in WW:
                for a in self.M.getActions(s, n):
                    self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
                    #self.M.Q[n][s][a] = (1.0 - self.alpha)*self.M.Q_bu[n][s][a] + self.alpha*self.M.Q[n][s][a]
                    self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]











 #####################################################################################

    def draw(self, weights):

        norm = sum(weights)
        ww = []
        for w in weights:
            ww.append(w / norm)
        ra = np.random.choice([0,1,2,3], p=ww)

        return ra
        # choice = np.random.uniform(0, sum(weights))
        # choiceIndex = 0
        #
        # for weight in weights:
        #     choice -= weight
        #     if choice <= 0:
        #         return choiceIndex
        #
        #     choiceIndex += 1

    def distr(self, weights, gamma=0.0):
        theSum = float(sum(weights))
        return list((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

    def mean(self, aList):
        theSum = 0
        count = 0

        for x in aList:
            theSum += x
            count += 1

        return 0 if count == 0 else theSum / count


    def exp3Update(self, n, currentState, numActions, reward, gamma, t, rewardMin = -1.0, rewardMax = 10.0):



        verbose = False
        if verbose:
            print("Iteration: ", t)
            print("Current state: ", currentState)

        weightsA = {}

        # Get weights for actions [UP, RIGHT, DOWN, LEFT]
        for a in sorted(self.M.weights[n][currentState].keys()):
            weightsA[a] = self.M.weights[n][currentState][a]

        # Put weights into a list
        weights = []
        for a in weightsA.keys():
            weights.append(weightsA[a])

        if verbose: print("Weights: ", weights)



        gamma = 0.5
        # Get distribution
        probabilityDistribution = self.distr(weights, gamma)

        # Loop thru and set pi
        aa = 0
        for a in sorted(self.M.weights[0][currentState].keys()):
            self.M.pi[0][currentState][a] = probabilityDistribution[aa]
            self.M.pi_sums[0][currentState][a] += self.M.pi[0][currentState][a]
            aa += 1

        # Get action choice from prob dist
        if verbose: print("ProbDist: ", probabilityDistribution)
        choice = self.draw(probabilityDistribution)

        # Get reward (Q-Value)
        if verbose: print("Choice: ", choice)  #0-3
        theReward = self.M.Q[0][currentState][choice]

        # Scale reward (Not sure this is correct, scaling is set in function arguments above) line 386
        if verbose: print("Reward for choice: ", theReward)
        scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin)  # rewards scaled to 0,1

        # Not entirely sure what this does
        if verbose: print("Scaled Reward: ", scaledReward)
        estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
        if verbose: print("Estimated reward: ", estimatedReward)

        # Update weights
        if verbose: print("Updating choice: ", choice)
        self.M.weights[0][currentState][choice] *= math.exp(estimatedReward * gamma / float(numActions))  # important that we use estimated reward here!
        if verbose: print("Weights: ", self.M.weights[0][currentState])
        if verbose: print("")

        # Return action
        return choice




######################################################################################

    def regretUpdate(self, n, currentState, t):

        iters = t + 1
        alphaR = 3.0 / 2.0  # accum pos regrets
        betaR = 0.0  # accum neg regrets
        gammaR = 2.0  # contribution to avg strategy

        alphaW = pow(iters, alphaR)
        alphaWeight = (alphaW / (alphaW + 1))

        betaW = pow(iters, betaR)
        betaWeight = (betaW / (betaW + 1))

        gammaWeight = pow((iters / (iters + 1)), gammaR)


        # Calculate regret - this variable name needs a better name
        target = 0.0

        # Skip terminal states
        if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            return

        for a in self.M.getActions(currentState, n):
            target += self.M.Q[n][self.M.getStateRep(currentState)][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]

        for a in self.M.getActions(currentState, n):
            action_regret = self.M.Q[n][self.M.getStateRep(currentState)][a] - target

            if self.DCFR:
                if action_regret > 0:
                    action_regret *= alphaWeight
                else:
                    action_regret *= betaWeight

            #RMPLUS = False
            if self.RMPLUS:
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            else:
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret

        # Skip terminal states
        # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
        #     continue

        for a in self.M.getActions(self.M.getStateRep(currentState), n):

            if self.M.isTerminal(currentState):
                continue
            # Sum up total regret
            rgrt_sum = 0.0
            for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
                rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0

            # Check if this is a "trick"
            # Check if this can go here or not
            if rgrt_sum > 0:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
            else:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(self.M.getStateRep(currentState), n))

            # Add to policy sum
            if self.DCFR:
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] * gammaWeight
            else:
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]
            #print("Pisumes: ", self.M.pi_sums[n][self.M.getStateRep(currentState)][a])

    def _lonr_value_iteration(self, t):


        WW = None
        # Q Update
        # All except Tiger Game
        # if self.randomize == False:
        for n in range(self.M.N):

            # Loop through all states
            for s in self.M.getStates():

                # Loop through actions of current player n
                # for a in self.M.getActions(s, n):
                a = self.M.getActions(self.M.getStateRep(s), n)
                if self.randomize == False:
                    self.QUpdate(n, s, a, randomS=None)
                else:
                    self.QUpdate(n, s=s, a_current2=a, randomS=s)

        # Tiger Game
        # elif self.randomize == True:
        #     print("Tiger game - value iteration")
        #     r = np.random.randint(0, 2)
        #     #if r == 0:
        #     if self.alt == 1:
        #         WW = self.M.totalStatesLeft
        #         self.alt = 2
        #     else:
        #         WW = self.M.totalStatesRight
        #         self.alt = 1
        #
        #     print("WW: ")
        #     print(WW)
        #     for n in range(self.M.N):
        #
        #         # Loop through all states
        #         for s in WW: #self.M.getStates(): #WW:
        #
        #             print(" S in loop: ", s)
        #             # Loop through actions of current player n
        #             # for a in self.M.getActions(s, n):
        #             aa = self.M.getActions(s, n)
        #             # if self.randomize == False:
        #             #     self.QUpdate(n, s, a, randomS=None)
        #             # else:
        #             print("AA: ", aa)
        #             for a in aa:
        #                 #print(" a: ", a)
        #                 al = []
        #                 al.append(a)
        #                 print(" a: ", a)
        #                 self.QUpdate(n, s=s, a_current2=al, randomS=s)

        # Q Backup
        for n in range(self.M.N):
            self.QBackup(n) #, WW=WW)

        # Update regret sums, pi, pi sums

        if self.randomize == False:
            for n in range(self.M.N):
                for s in self.M.getStates():
                    self.regretUpdate(n, s, t)
        else:
            for n in range(self.M.N):
                for s in self.M.getStates(): #WW: #self.M.totalStatesLeft: #WW:
                    if self.M.isTerminal(s): continue
                    self.regretUpdate(n, s, t)





    def lonr_online(self, iterations=-1, log=-1, randomized=False):

        print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha)

            self._lonr_online(t=t, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            self.iters.append(t)

            if self.GRIDWORLD:
                if self.M.noise > 0:
                    self.runningSum += self.M.Q[0][36][3]
                else:
                    self.runningSum += self.M.Q[0][36][0]
                self.QSums.append(float(self.runningSum) / float(t))
            elif self.NOSDE:
                # Player 0, state left, KEEP
                self.QSumSendRunningSum += self.M.Q[0][1]["SEND"]
                self.QSumSEND.append(float(self.QSumSendRunningSum) / float(t))
                # Player 0, state left
                self.QSumKEEPRunningSum += self.M.Q[0][1]["KEEP"]
                self.QSumKEEP.append(float(self.QSumKEEPRunningSum) / float(t))

                self.QSumNOOPRunningSum += self.M.Q[0][2]["NOOP"]
                self.QSumNOOP.append(float(self.QSumNOOPRunningSum) / float(t))


            elif self.SOCCER:
                # print("VAL: ", self.M.Q[0]["B21"])
                self.QSumSoccerB21RightRunningSum += self.M.Q[0]["B21"][3]
                self.QSumSoccerB21Right.append(float(self.QSumSoccerB21RightRunningSum) / float(t))

            elif self.TIGERGAME:
                self.LRunningSum += self.M.Q[0]["rootTL"]["L"]
                self.L.append(float(self.LRunningSum) / float(t))
                self.OLRunningSum += self.M.Q[0]["rootTL"]["OL"]
                self.OL.append(float(self.OLRunningSum) / float(t))
                self.ORRunningSum += self.M.Q[0]["rootTL"]["OR"]
                self.OR.append(float(self.ORRunningSum) / float(t))

                self.rootTLLGLLGLRS += self.M.Q[0]["rootTLLGLLGL"]["OR"]
                self.rootTLLGLLGL.append(float(self.rootTLLGLLGLRS) / float(t))

                self.rootTLLGRLGRRS += self.M.Q[0]["rootTLLGRLGR"]["OL"]
                self.rootTLLGRLGR.append(float(self.rootTLLGRLGRRS) / float(t))

                self.rootTLLGLLGRRS += self.M.Q[0]["rootTLLGLLGR"]["L"]
                self.rootTLLGLLGR.append(float(self.rootTLLGLLGRRS) / float(t))

                self.rootTLLGRLGLRS += self.M.Q[0]["rootTLLGRLGL"]["L"]
                self.rootTLLGRLGL.append(float(self.rootTLLGRLGLRS) / float(t))



            tt = t

            # print("Finish Training")
            # print("LastAVG: ", float(self.runningSum) / float(tt))
            # print("Avg: ", float(self.runningSum) / float(iterations))
            # print(self.QSumSEND)
            # print(self.QSums)
            # print("PRINTED QSUMS")
        if self.GRIDWORLD:
            return self.QSums, self.iters
        elif self.NOSDE:

            return self.QSumSEND, self.QSumKEEP, self.QSumNOOP, self.iters
        elif self.SOCCER:
            return self.QSumSoccerB21Right, self.iters

        elif self.TIGERGAME:
            print("PISUMS:")
            print("PISUM LGLLGL: ", self.M.pi_sums[0]["rootTLLGLLGL"])
            print("PISUM LGRLGR: ", self.M.pi_sums[0]["rootTLLGRLGR"])
            print("PISUM LGLLGR: ", self.M.pi_sums[0]["rootTLLGLLGR"])
            print("PISUM LGRLGL: ", self.M.pi_sums[0]["rootTLLGRLGL"])
            print("")
            # print("Q:")
            # print("PISUM LGLLGL: ", self.M.Q[0]["rootTLLGLLGL"])
            # print("PISUM LGRLGR: ", self.M.Q[0]["rootTLLGRLGR"])
            # print("PISUM LGLLGR: ", self.M.Q[0]["rootTLLGLLGR"])
            # print("PISUM LGRLGL: ", self.M.Q[0]["rootTLLGRLGL"])

            print("Final Q of GRGROL: ", self.rootTLLGRLGR[-1])

            return self.L, self.OL, self.OR, self.rootTLLGLLGL, self.rootTLLGRLGR, self.rootTLLGLLGR, self.rootTLLGRLGL, self.iters

        print("Finish Training")


    def _lonr_online(self, t=0, totalIterations=-1, randomized=False):
        """

        :param t: current iteration
        :param totalIterations: total iterations
        :param randomized: Randomizes the start state.
                        True: For Tiger Game, to switch between TigerOnLeft, TigerOnRight
                        False: For GridWorld, only one start state.
        :return:
        """


        # For Tiger Game, randomize the MDP it sees
        if randomized:

            # Get the top root
            startStates = self.M.getNextStatesAndProbs(self.M.startState, None, 0)

            totalStartStates = []
            totalStartStateProbs = []

            # Pick Tiger on Left/Right based on probability set in M
            for nextState, nextStateProb, _ in startStates:
                totalStartStates.append(nextState)
                totalStartStateProbs.append(nextStateProb)

            # Randomly pick TigerOnLeft/Right based on M.TLProb (1-M.TLProb = TRProb)
            currentState = np.random.choice(totalStartStates, p=totalStartStateProbs)

        # For GridWorld, set currentState to the startState
        else:

            currentState = self.M.startState

        #print("RANDOMIZED: ", randomized)
        done = False
        n = 0  # One player

        #print("")
        # Episode loop - until terminal state is reached
        while done == False:

            #print("Current state: ", currentState)
            if self.M.isTerminal(currentState) == True:
                # for a in self.M.getActions(currentState, 0):
                #     # Note, the reward is via the actual state, so there is no getStateRep()
                #     self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * self.M.getReward(currentState,a,a,a)
                a = self.M.getActions(currentState, 0)
                self.QUpdate(n, currentState, a, currentState)
                done = True
                continue

            a = self.M.getActions(currentState, 0)
            self.QUpdate(n, currentState, a, currentState)
            # totStates keeps track of which states need Qvalue copying
            #   as not all states need to be backed up, only the ones visited
            totStates = []
            totStates.append(currentState)

            # Q Backup
            for n in range(self.M.N):
                self.QBackup(n)


            # Don't update terminal states
            if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
                done = True
                continue

            if self.EXP3 == False:
                self.regretUpdate(n, self.M.getStateRep(currentState), t)

                # Epsilon Greedy action selection
                if np.random.randint(0, 100) < int(self.epsilon):
                    totalActions = self.M.getActions(currentState, 0)
                    randomAction = np.random.randint(0, len(totalActions))
                    randomAction = totalActions[randomAction]

                else:

                    totalActions = []
                    totalActionsProbs = []
                    ta = self.M.getActions(currentState, 0)
                    for action in ta:
                        totalActions.append(action)
                        totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])
                    #print(totalActionsProbs)
                    randomAction = np.random.choice(totalActions, p=totalActionsProbs)

            #Exp 3 is true
            else:

                randomAction = self.exp3Update(n, currentState, 4, 0, 0.5, t)
            # randomAction picked, now simulate taking action
            #       GridWorld: This will handle non-determinism, if there is non-determinism
            #       TigerGame: This will either get the one next state OR
            #                       if action is LISTEN, it will return next state based on
            #                       observation accuracy aka (85/15) or (15/85), which is
            #                       equivalent to hearing a growl left or growl right
            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)

            #print("NPS: ", nextPossStates)
            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                currentState = nextPossStates[0][0]

            # More than one possible successor, pick via probabilities
            #       GridWorld non-determ:
            #           3 states: 1-self.noise, (1-self.noise)/2, (1-self.noise)/2
            #               ex: 0.8, 0.1, 0.1 for randomAction, to the side, to the side
            #
            #       Tiger Game:
            #           Only happens when randomAction is Listen
            #               Return is based on which side tiger is on, and the obs accuracy
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp,_ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)
                currentState = np.random.choice(nextStates, p=nextStateProbs)

            #print("BCS: ", currentState)
            # if self.M.isTerminal(currentState):
            #     done = True





















            # Loop through all actions
            # for a in self.M.getActions(currentState, 0):
            #
            #     # Get successor states
            #     succs = self.M.getNextStatesAndProbs(currentState, a, 0)
            #
            #     Value = 0.0
            #     # Loop through each s', T(s' | s, a)
            #     #   - Last parameter in loop is reward, which eventually will be included
            #     #       -
            #     for s_prime, prob, _ in succs:
            #
            #         tempValue = 0.0
            #         for a_prime in self.M.getActions(s_prime, 0):
            #             tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_prime]
            #         Value += prob * (self.M.getReward(currentState, a, 0,0) + self.gamma * tempValue)
            #
            #         if s_prime not in totStates:
            #             totStates.append(s_prime)
            #
            #     self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * Value


            # Copy Q Values over for states visited
            # for s in totStates:
            #     for a in self.M.getActions(s, 0):
            #         self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
            #         # self.Qsums[self.tigergame.getStateRep(s)][a] += self.Q_bu[self.tigergame.getStateRep(s)][a]

###################################################################################

# Calculate regret - this variable name needs a better name
            # target = 0.0
            #
            # for a in self.M.getActions(currentState, 0):
            #     target += self.M.Q[n][self.M.getStateRep(currentState)][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]
            #
            # for a in self.M.getActions(currentState, 0):
            #     action_regret = self.M.Q[n][self.M.getStateRep(currentState)][a] - target
            #
            #     #RMPLUS = True
            #     if self.RMPLUS:
            #         self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            #     else:
            #         self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret
            #
            #
            # # Skip terminal states
            # # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            # #     continue
            #
            # for a in self.M.getActions(self.M.getStateRep(currentState), 0):
            #
            #     # Sum up total regret
            #     rgrt_sum = 0.0
            #     for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
            #         rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0
            #
            #     # Check if this is a "trick"
            #     # Check if this can go here or not
            #     if rgrt_sum > 0:
            #         self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
            #     else:
            #         self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(self.M.getStateRep(currentState),0))
            #
            #     # Add to policy sum
            #     self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]


#####################################################################################################




# if self.M.isTerminal(s): return True
        # if self.M.isTerminal(s):
        #     self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha) * self.M.Q[n][s][a_current] + self.alpha * self.M.getReward(s, a_current, 0, 0)
        #     continue
        #
        # # Get next states and transition probs for MDP
        # # Get next states and other players policy for MG
        # succs = self.M.getNextStatesAndProbs(s, a_current, n)
        # # print("s: ", s, " a:", a_current, "   succs: ", succs)
        #
        # # print("  SUCCS: ", succs)
        # Value = 0.0
        #
        # # Loop thru each next state and prob
        # for s_prime, prob, reward in succs:
        #     # print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
        #     tempValue = 0.0
        #
        #     # Loop thru actions in s_prime for player n
        #     for a_current_prime in self.M.getActions(s_prime, n):
        #         tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][s_prime][a_current_prime]
        #
        #     Value += prob * (reward + self.gamma * tempValue)
        #
        # self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha) * self.M.Q[n][s][a_current] + (self.alpha) * Value

        # print("Entered QUpdate: ", " s: ", currentState, " a:", a_current)
        # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
        #     self.M.Q_bu[n][self.M.getStateRep(currentState)][a_current] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a_current] + self.alpha * self.M.getReward(self.M.getStateRep(currentState), a_current, 0, 0)
        #     #for a in self.M.getActions(currentState, n):
        #         # Note, the reward is via the actual state, so there is no getStateRep()
        #     #    self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * self.M.getReward(currentState, a, a, a)
        #     return
        #     # done = True
        #     # continue
        #
        # # totStates keeps track of which states need Qvalue copying
        # #   as not all states need to be backed up, only the ones visited
        # totStates = []
        # totStates.append(currentState)
        #
        # # Loop through all actions
        # for a in self.M.getActions(currentState, n):
        #     # print(" - in loop a: ", a)
        #     # Get successor states
        #     succs = self.M.getNextStatesAndProbs(currentState, a, n)
        #
        #     Value = 0.0
        #     # Loop through each s', T(s' | s, a)
        #     #   - Last parameter in loop is reward, which eventually will be included
        #     #       -
        #     for s_prime, prob, _ in succs:
        #
        #         tempValue = 0.0
        #         for a_prime in self.M.getActions(s_prime, n):
        #             # print("  -- SR: ", self.M.getStateRep(s_prime))
        #             # print("    - ", self.M.Q[n][self.M.getStateRep(s_prime)]) #self.M.pi[n][self.M.getStateRep(s_prime)][a_prime])
        #             tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_prime]
        #         Value += prob * (self.M.getReward(self.M.getStateRep(currentState), a, 0, 0) + self.gamma * tempValue)
        #
        #         # REMOVE: Q[S_PRIME] is not updated/changed
        #         if s_prime not in totStates:
        #             totStates.append(s_prime)
        #
        #     self.M.Q_bu[n][self.M.getStateRep(currentState)][a_current] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a_current] + self.alpha * Value

        # Copy Q Values over for states visited
        # for s in totStates:
        #     for a in self.M.getActions(s, n):
        #         self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
                # self.Qsums[self.tigergame.getStateRep(s)][a] += self.Q_bu[self.tigergame.getStateRep(s)][a]

        # Don't update terminal states
        # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
        #     continue

#############################################################################


 #             if self.M.isTerminal(s):
        #                 self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] + self.alpha*self.M.getReward(s, a_current, 0,0)
        #                 continue
        #
        #             # Get next states and transition probs for MDP
        #             # Get next states and other players policy for MG
        #             succs = self.M.getNextStatesAndProbs(s, a_current, n)
        #             #print("s: ", s, " a:", a_current, "   succs: ", succs)
        #
        #             #print("  SUCCS: ", succs)
        #             Value = 0.0
        #
        #             #Loop thru each next state and prob
        #             for s_prime, prob, reward in succs:
        #                 #print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
        #                 tempValue = 0.0
        #
        #                 # Loop thru actions in s_prime for player n
        #                 for a_current_prime in self.M.getActions(s_prime, n):
        #                     tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][s_prime][a_current_prime]
        #
        #                 Value += prob * (reward + self.gamma * tempValue)
        #
        #             self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] + (self.alpha)*Value
        #
        #
        # ####################################################################################################
        #
        # Back up Q
        # for n in range(self.M.N):
        #     for s in self.M.getStates():
        #         for a in self.M.getActions(s, n):
        #             self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
        #             self.M.QSums[n][s][a] += self.M.Q[n][s][a]

        # Regrets / policy updates, etc
        # For each player

        # iters = t + 1
        # alphaR = 3.0 / 2.0  # accum pos regrets
        # betaR = 0.0         # accum neg regrets
        # gammaR = 2.0        # contribution to avg strategy
        #
        # alphaW = pow(iters, alphaR)
        # alphaWeight = (alphaW / (alphaW + 1))
        #
        # betaW = pow(iters, betaR)
        # betaWeight = (betaW / (betaW + 1))
        #
        # gammaWeight = pow((iters / (iters + 1)), gammaR)
        #
        # for n in range(self.M.N):
        #
        #     # For each state
        #     for s in self.M.getStates():
        #
        #         # Skip terminals
        #         if self.M.isTerminal(s): continue
        #
        #         target = 0.0
        #         for a in self.M.getActions(s, n):
        #             target += self.M.Q[n][s][a] * self.M.pi[n][s][a]
        #
        #         for a in self.M.getActions(s, n):
        #             action_regret = self.M.Q[n][s][a] - target
        #
        #             if self.DCFR:
        #                 if action_regret > 0:
        #                     action_regret *= alphaWeight
        #                 else:
        #                     action_regret *= betaWeight
        #
        #             self.M.regret_sums[n][s][a] += action_regret
        #             #self.M.regret_sums[n][s][a] = max(0.0, self.M.regret_sums[n][s][a] + action_regret)
        #
        #     for s in self.M.getStates():
        #
        #         # Skip terminals
        #         if self.M.isTerminal(s): continue
        #
        #         rgrt_sum = 0.0
        #         for a in self.M.getActions(s, n):
        #             rgrt_sum += max(0.0, self.M.regret_sums[n][s][a])
        #
        #         for a in self.M.getActions(s, n):
        #             if rgrt_sum > 0:
        #                 self.M.pi[n][s][a] = (max(self.M.regret_sums[n][s][a], 0.0) / rgrt_sum)
        #             else:
        #                 self.M.pi[n][s][a] = 1. / len(self.M.getActions(s, n))
        #
        #             if self.DCFR:
        #                 self.M.pi_sums[n][s][a] += self.M.pi[n][s][a] * gammaWeight
        #             else:
        #                 self.M.pi_sums[n][s][a] += self.M.pi[n][s][a]









##################################




#
# import numpy as np
#
#
# class MDP(object):
#
#     def __init__(self, startState=None):
#
#         # Check if these are used
#         self.totalPlayers = 1
#         self.totalStates = 1
#
#         # Initialized here but must be defined/created in created class
#         self.Q = None
#         self.Q_bu = None
#         self.QSums = None
#         self.pi = None
#         self.pi_sums = None
#         self.regret_sums = None
#
#         self.startState = startState
#
#     def getActions(self, s, n):
#         """ Returns list of actions
#
#         """
#         raise NotImplementedError("Please implement information_set method")
#
#     def getNextStatesAndProbs(self, s, a_current, n_current):
#         """ Returns list of [ [next_state, prob, rew] ]
#
#         """
#         raise NotImplementedError("Please implement information_set method")
#
#
#     def getReward(self, s, a_current, n, a_notN):
#         """ s=state, a_current=actionOfPlayer n, n is current player, a_notN is action of other player
#             Joint Actions required for Markov Games
#             a_notN can be set to 0 for MDPs and ignored
#
#         Returns reward(s, a, a_other)
#
#         """
#         raise NotImplementedError("Please implement information_set method")
#
#     def getStates(self):
#         """Return List of States
#
#         """
#         raise NotImplementedError("Please implement information_set method")
#
#     def isTerminal(self, s):
#         raise NotImplementedError("Please implement information_set method")
#
#
#     # The following are only needed for O-LONR
#
#     def getMove(self, s, a):
#         """Used to get actual move in the environment for O-LONR
#         """
#         raise NotImplementedError("Please implement information_set method")
#
#     def getStateRep(self, s):
#         """ O-LONR use only
#
#             For MDPs: simply returns state
#             For TigerGame: Maps both to one (EX: TLL & TRL -> TLL) aka one set of Q Values
#         """
#         raise NotImplementedError("Please implement information_set method")
#
# class LONR(object):
#
#     # M: Markov game (MDP, markov game, tiger game)
#
#     def __init__(self, M=None, gamma=1.0, alpha=1.0, epsilon=10, alphaDecay=1.0, DCFR=False):
#         self.M = M
#         #self.N = M.totalPlayers
#         #self.S = M.totalStates
#
#         self.gamma = gamma
#         self.alpha = alpha
#         self.epsilon = epsilon
#         self.alphaDecay = alphaDecay
#
#
#         self.DCFR = DCFR
#
#     def lonr_value_iteration(self, iterations=-1, log=-1):
#
#         print("Starting training..")
#         for t in range(1, iterations+1):
#
#             if (t+1) % log == 0:
#                 print("Iteration: ", t+1, " alpha: ", self.alpha)
#
#             self._lonr_value_iteration(t=t)
#             #print("")
#             self.alpha *= self.alphaDecay
#             self.alpha = max(0.0, self.alpha)
#
#         print("Finish Training")
#
#     def _lonr_value_iteration(self, t):
#
#         # Loop through all players
#         for n in range(self.M.N):
#             #print("N: ", n)
#
#             # Loop through all states
#             for s in self.M.getStates():
#                 #print(" s: ", s)
#
#                 # Loop through actions of current player n
#                 for a_current in self.M.getActions(s, n):
#                     #print("  a_current: ", a_current)
#
#                     if self.M.isTerminal(s):
#                         # This is here because terminals have actions like "exit"
#                         # to collect the reward. To make the code easier, the terminals
#                         # have the same actions as everywhere else, but each action is an "exit"
#                         #
#                         # This was a remnant of when I was using numpy arrays and couldn't set the size
#                         # for one index different than the rest easily (Q[totalStates][totalActions] for
#                         # instance, but Q[terminals][oneExitAction]
#                         #
#                         # But now the MDP class handles that, and if I change them all to dicts/maps
#                         # and not numpy arrays, I wont need this special case here.
#                         #
#                         # This works out below if/when an expectation is taken because the initial
#                         # uniform policy never changes so the expected value = reward
#                         #
#                         # I might possibly change it so that instead of setting it here,
#                         # M.getActions(terminal) returns one exit action
#                         self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] * + self.alpha*self.M.getReward(s, a_current, 0,0)
#                         #print("Terminal: ", s)
#                         continue
#
#                     # Get next states and transition probs for MDP
#                     # Get next states and other players policy for MG
#                     succs = self.M.getNextStatesAndProbs(s, a_current, n)
#                     #print("s: ", s, " a:", a_current, "   succs: ", succs)
#
#
#                     Value = 0.0
#
#                     #Loop thru each next state and prob
#                     for s_prime, prob, reward in succs:
#                         #print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
#                         tempValue = 0.0
#
#                         # Loop thru actions in s_prime for player n
#                         for a_current_prime in self.M.getActions(s_prime, n):
#                             tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][s_prime][a_current_prime]
#
#                         Value += prob * (reward + self.gamma * tempValue)
#
#                     self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] + (self.alpha)*Value
#
#
#         ####################################################################################################
#
#         # Back up Q
#         for n in range(self.M.N):
#             for s in self.M.getStates():
#                 for a in self.M.getActions(s, n):
#                     self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
#                     self.M.QSums[n][s][a] += self.M.Q[n][s][a]
#
#         # Regrets / policy updates, etc
#         # For each player
#
#         iters = t + 1
#         alphaR = 3.0 / 2.0  # accum pos regrets
#         betaR = 0.0         # accum neg regrets
#         gammaR = 2.0        # contribution to avg strategy
#
#         alphaW = pow(iters, alphaR)
#         alphaWeight = (alphaW / (alphaW + 1))
#
#         betaW = pow(iters, betaR)
#         betaWeight = (betaW / (betaW + 1))
#
#         gammaWeight = pow((iters / (iters + 1)), gammaR)
#
#         # Update regret sums
#         for n in range(self.M.N):
#
#             # For each state
#             for s in self.M.getStates():
#
#                 # Skip terminals
#                 if self.M.isTerminal(s): continue
#
#                 target = 0.0
#                 for a in self.M.getActions(s, n):
#                     target += self.M.Q[n][s][a] * self.M.pi[n][s][a]
#
#                 for a in self.M.getActions(s, n):
#                     action_regret = self.M.Q[n][s][a] - target
#
#                     if self.DCFR:
#                         if action_regret > 0:
#                             action_regret *= alphaWeight
#                         else:
#                             action_regret *= betaWeight
#
#                     self.M.regret_sums[n][s][a] += action_regret
#                     #self.M.regret_sums[n][s][a] = max(0.0, self.M.regret_sums[n][s][a] + action_regret)
#
#             # Update pi and pi sums
#             for s in self.M.getStates():
#
#                 # Skip terminals
#                 if self.M.isTerminal(s): continue
#
#                 rgrt_sum = 0.0
#                 for a in self.M.getActions(s, n):
#                     rgrt_sum += max(0.0, self.M.regret_sums[n][s][a])
#
#                 for a in self.M.getActions(s, n):
#                     if rgrt_sum > 0:
#                         self.M.pi[n][s][a] = (max(self.M.regret_sums[n][s][a], 0.0) / rgrt_sum)
#                     else:
#                         self.M.pi[n][s][a] = 1. / len(self.M.getActions(s, n))
#
#                     if self.DCFR:
#                         self.M.pi_sums[n][s][a] += self.M.pi[n][s][a] * gammaWeight
#                     else:
#                         self.M.pi_sums[n][s][a] += self.M.pi[n][s][a]
#
#
#     def lonr_online(self, iterations=-1, log=-1, randomized=False):
#
#         print("Starting training..")
#         for t in range(1, iterations+1):
#
#             if (t+1) % log == 0:
#                 print("Iteration: ", t+1, " alpha:", self.alpha)
#
#             self._lonr_online(t=t, totalIterations=iterations, randomized=randomized)
#
#
#
#             self.alpha *= self.alphaDecay
#             self.alpha = max(0.0, self.alpha)
#
#         print("Finish Training")
#
#
#     def _lonr_online(self, t=0, totalIterations=-1, randomized=False):
#         """
#
#         :param t: current iteration
#         :param totalIterations: total iterations
#         :param randomized: Randomizes the start state.
#                         True: For Tiger Game, to switch between TigerOnLeft, TigerOnRight
#                         False: For GridWorld, only one start state.
#         :return:
#         """
#
#
#         # For Tiger Game, randomize the MDP it sees
#         if randomized:
#
#             # Get the top root
#             startStates = self.M.getNextStatesAndProbs(self.M.startState, None, 0)
#
#             totalStartStates = []
#             totalStartStateProbs = []
#
#             # Pick Tiger on Left/Right based on probability set in M
#             for nextState, nextStateProb, _ in startStates:
#                 totalStartStates.append(nextState)
#                 totalStartStateProbs.append(nextStateProb)
#
#             # Randomly pick TigerOnLeft/Right based on M.TLProb (1-M.TLProb = TRProb)
#             currentState = np.random.choice(totalStartStates, p=totalStartStateProbs)
#
#         # For GridWorld, set currentState to the startState
#         else:
#             currentState = self.M.startState
#
#
#         done = False
#         n = 0  # One player
#
#         # Episode loop - until terminal state is reached
#         while done == False:
#
#             # Terminal - Set Q as Reward (ALL terminals have one action - "exit" which receives reward)
#             # Note:
#             #       GridWorld Terminals have the same 4 actions.
#             #           All get set here to the full reward
#             #           pi for each action remains 1 / totalActions
#             #       I intend to remove this part and include terminal rewards below in successors
#             #       But for now, in the expectation taken below, the value is correct:
#             #           For example:
#             #               Q[term] = (1/4)Reward + (1/4)Reward + (1/4)Reward + (1/4)Reward = Reward
#             #
#             #       Also, this backup is split via self.alpha.
#             #           For GridWorld, this does nothing (self.alpha = 1.0, removes the first part)
#             #           For Tiger Game, this allows shared Q's to converge and not be overwritten
#             if self.M.isTerminal(currentState) == True:
#                 for a in self.M.getActions(currentState, 0):
#                     # Note, the reward is via the actual state, so there is no getStateRep()
#                     self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * self.M.getReward(currentState,a,a,a)
#                 done = True
#                 continue
#
#
#             # totStates keeps track of which states need Qvalue copying
#             #   as not all states need to be backed up, only the ones visited
#             totStates = []
#             totStates.append(currentState)
#
#             # Loop through all actions
#             for a in self.M.getActions(currentState, 0):
#
#                 # Get successor states
#                 succs = self.M.getNextStatesAndProbs(currentState, a, 0)
#
#                 Value = 0.0
#                 # Loop through each s', T(s' | s, a)
#                 #   - Last parameter in loop is reward, which eventually will be included
#                 #       -
#                 for s_prime, prob, _ in succs:
#
#                     tempValue = 0.0
#                     for a_prime in self.M.getActions(s_prime, 0):
#                         tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_prime]
#                     Value += prob * (self.M.getReward(currentState, a, 0,0) + self.gamma * tempValue)
#
#                     if s_prime not in totStates:
#                         totStates.append(s_prime)
#
#                 self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * Value
#
#
#             # if max(list(self.M.Q[0][36].values())) -  max(list(self.M.Q_bu[0][36].values())) != 0:
#             #     print("Difference: ", max(list(self.M.Q[0][36].values())) -  max(list(self.M.Q_bu[0][36].values())))
#
#             # Copy Q Values over for states visited
#             for s in totStates:
#                 for a in self.M.getActions(s, 0):
#                     self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
#                     # self.Qsums[self.tigergame.getStateRep(s)][a] += self.Q_bu[self.tigergame.getStateRep(s)][a]
#
#
#             # Don't update terminal states
#             # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
#             #     continue
#
#             # Calculate regret - this variable name needs a better name
#             target = 0.0
#
#             for a in self.M.getActions(currentState, 0):
#                 target += self.M.Q[n][self.M.getStateRep(currentState)][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]
#
#             for a in self.M.getActions(currentState, 0):
#                 action_regret = self.M.Q[n][self.M.getStateRep(currentState)][a] - target
#
#                 RMPLUS = True
#                 if RMPLUS:
#                     self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
#                 else:
#                     self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret
#
#
#             # Skip terminal states
#             # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
#             #     continue
#
#             for a in self.M.getActions(self.M.getStateRep(currentState), 0):
#
#                 # Sum up total regret
#                 rgrt_sum = 0.0
#                 for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
#                     rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0
#
#                 # Check if this is a "trick"
#                 # Check if this can go here or not
#                 if rgrt_sum > 0:
#                     self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
#                 else:
#                     self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(self.M.getStateRep(currentState),0))
#
#                 # Add to policy sum
#                 self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]
#
#             # Epsilon Greedy action selection
#             if np.random.randint(0, 100) < int(self.epsilon):
#                 totalActions = self.M.getActions(currentState, 0)
#                 randomAction = np.random.randint(0, len(totalActions))
#                 randomAction = totalActions[randomAction]
#
#             else:
#
#                 totalActions = []
#                 totalActionsProbs = []
#                 ta = self.M.getActions(currentState, 0)
#                 for action in ta:
#                     totalActions.append(action)
#                     totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])
#                 #print(totalActionsProbs)
#                 randomAction = np.random.choice(totalActions, p=totalActionsProbs)
#
#             # randomAction picked, now simulate taking action
#             #       GridWorld: This will handle non-determinism, if there is non-determinism
#             #       TigerGame: This will either get the one next state OR
#             #                       if action is LISTEN, it will return next state based on
#             #                       observation accuracy aka (85/15) or (15/85), which is
#             #                       equivalent to hearing a growl left or growl right
#             nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
#
#             # If there is only one successor state, pick that
#             if len(nextPossStates) == 1:
#                 # nextPossStates is list of lists
#                 # nextPossStates = [[next_state, prob, reward]]
#                 currentState = nextPossStates[0][0]
#
#             # More than one possible successor, pick via probabilities
#             #       GridWorld non-determ:
#             #           3 states: 1-self.noise, (1-self.noise)/2, (1-self.noise)/2
#             #               ex: 0.8, 0.1, 0.1 for randomAction, to the side, to the side
#             #
#             #       Tiger Game:
#             #           Only happens when randomAction is Listen
#             #               Return is based on which side tiger is on, and the obs accuracy
#             else:
#                 nextStates = []
#                 nextStateProbs = []
#                 for ns, nsp,_ in nextPossStates:
#                     nextStates.append(ns)
#                     nextStateProbs.append(nsp)
#                 currentState = np.random.choice(nextStates, p=nextStateProbs)
#
