from LONRGeneral.LONR import *
import copy
import numpy as np


class SoccerGame(MDP):

    def __init__(self):
        super().__init__()

        self.N = 2

        self.rows = 2
        self.cols = 4
        self.num_states = self.rows * self.cols


        self.total_states = self.create_state_comb(list(range(self.num_states)), list(range(self.num_states)))
        self.total_actions = [0, 1, 2, 3]

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
                self.pi[n][s] = {}
                self.regret_sums[n][s] = {}
                self.pi_sums[n][s] = {}
                for a in self.total_actions:
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.pi[n][s][a] = 1.0 / 4.0 #len(list(total_actions.keys())
                    self.regret_sums[n][s][a] = 0.0
                    self.pi_sums[n][s][a] = 0.0


    def getActions(self, s, n):
        return self.total_actions

    def getStates(self):
        return self.total_states

    def getStateRep(self, s):
        return s

    def getNextStatesAndProbs(self, s, a_current, n_current):
        """

        Returns list of lists: [[nextState, prob, reward]]


        """
        successors = []

        # Player with ball
        player_with_ball = s[0]

        pAball = False
        pBball = False
        if player_with_ball == "A":
            pAball = True
        if player_with_ball == "B":
            pBball = True

        playerApos = int(s[1])
        playerBpos = int(s[2])

        pAx, pAy = self.stateToXY(playerApos)
        pBx, pBy = self.stateToXY(playerBpos)

        actions_A = a_current

        otherN = 1 if n_current == 0 else 0

        # Returns list of all s_primes, other player's pi[s][a], and r(s, aA, aB)

        ################################
        # This is original for updating player A:
        #
        # for actions_A in self.total_actions:
        #     QV = 0.0
        #     oppValue = 0.0
        #     for actions_B in self.total_actions:
        #
        #         actions = {'A': actions_A, 'B': actions_B}
        #         new_state, rewards, goal = world.move(actions)
        #
        #
        #         rA = rewards['A']
        #
        #         Value = 0.0
        #
        #         for action_ in self.total_actions:
        #             Value += self.QvaluesA[new_state][action_] * self.piA[new_state][action_]
        #         QV += self.piB[s][actions_B] * (rA + self.gamma * Value)
        #
        #     self.QValue_backupA[s][actions_A] = (0.1 * self.QvaluesA[s][actions_A]) + (0.9 * QV)

        # So instead of a loop, each next_state, reward, and piB is returned
        ################################

        for actions_B in self.getActions(s, n_current): #should be other player's actions but since they have the same set..
            player_a = Player(x=pAx, y=pAy, has_ball=pAball, p_id='A')
            player_b = Player(x=pBx, y=pBy, has_ball=pBball, p_id='B')

            world = World()
            world.set_world_size(x=self.cols, y=self.rows)
            world.place_player(player_a, player_id='A')
            world.place_player(player_b, player_id='B')
            world.set_goals(100, 0, 'A')
            world.set_goals(100, 3, 'B')

            # if world.map_player_state() == "A25":
            #     print("State: ", world.map_player_state())
            #     world.plot_grid()

            if n_current == 0:
                actions = {'A': actions_A, 'B': actions_B}
            else:
                actions = {'A': actions_B, 'B': actions_A}
            new_state, rewards, goal = world.move(actions)
            reward = rewards['A'] if n_current == 0 else rewards['B']
            successors.append([new_state, self.pi[otherN][s][actions_B], reward])

        return successors

    
    def getReward(self, s, a_current, n, a_notN):
        """Returns reward(s, a, a_other)

        """
        return None
        #return 0.0


    def isTerminal(self, s):
        """Returns true is state is terminal

        """
        if s == "A46" or s == "B54" or s == "A06" or s == "A74":
            return True
        if s == "B04" or s == "B67" or s == "B70" or s == "A03":
            return True
        if s == "B60" or s == "B73" or s == "A37" or s == "A34":
            return True
        if s == "B27" or s == "B57" or s == "B50" or s == "B24":
            return True
        if s == "A40" or s == "A45" or s == "A72" or s == "B14":
            return True
        if s == "A31" or s == "B37" or s == "A75" or s == "B17":
            return True
        if s == "A32" or s == "A43" or s == "B13" or s == "B40":
            return True
        if s == "B74" or s == "A47" or s == "A30" or s == "B47":
            return True
        if s == "A71" or s == "A36" or s == "A07" or s == "B63":
            return True
        if s == "A01" or s == "B53" or s == "B34" or s == "A35":
            return True
        if s == "B23" or s == "B64" or s == "B43" or s == "A05":
            return True
        if s == "B07" or s == "B03" or s == "A73" or s == "A41":
            return True
        if s == "A42" or s == "B20" or s == "B10" or s == "A76":
            return True
        if s == "B30" or s == "A04" or s == "A70" or s == "A02":
            return True
        return False

    def stateToXY(self, stateID):
        """Convert state ID back into x,y coordinates

            Soccer grid is indexed by:

            0 1 2 3
            4 5 6 7

            x,y coordinates are:

            00 10 20 30
            01 11 21 31

        """

        x = 0
        y = 0
        if stateID == 0:
            x = 0
            y = 0
        elif stateID == 1:
            x = 1
            y = 0
        elif stateID == 2:
            x = 2
            y = 0
        elif stateID == 3:
            x = 3
            y = 0
        elif stateID == 4:
            x = 0
            y = 1
        elif stateID == 5:
            x = 1
            y = 1
        elif stateID == 6:
            x = 2
            y = 1
        elif stateID == 7:
            x = 3
            y = 1
        return x, y
        

    def normalize_pisums(self):
        for n in range(self.N):
            for k in self.pi_sums[n].keys():
                tot = 0.0
                for kk in self.pi_sums[n][k].keys():
                    tot += self.pi_sums[n][k][kk]
                tot = 1.0 if tot <= 0 else tot
                for kk in self.pi_sums[n][k].keys():
                    self.pi_sums[n][k][kk] = self.pi_sums[n][k][kk] / float(tot)
        
    def play(self, iterations=1, log=-1):
        playerARewards = 0.0
        playerBRewards = 0.0
        playerAWins = 0
        playerBWins = 0
        for t in range(iterations):

            if (t + 1) % log == 0:
                print("Test iteration: ", t + 1)

            player_a = Player(x=2, y=0, has_ball=True, p_id='A')
            player_b = Player(x=1, y=0, has_ball=False, p_id='B')

            world = World()
            world.set_world_size(x=self.cols, y=self.rows)
            world.place_player(player_a, player_id='A')
            world.place_player(player_b, player_id='B')
            world.set_goals(100, 0, 'A')
            world.set_goals(100, 3, 'B')

            curState = world.map_player_state()

            goal = False
            while not goal:
                piAA = [self.pi_sums[0][curState][0], self.pi_sums[0][curState][1], self.pi_sums[0][curState][2],
                        self.pi_sums[0][curState][3]]
                piBB = [self.pi_sums[1][curState][0], self.pi_sums[1][curState][1], self.pi_sums[1][curState][2],
                        self.pi_sums[1][curState][3]]

                # Pick action according to learn policy
                pAa = np.random.choice([0, 1, 2, 3], p=piAA)
                pBa = np.random.choice([0, 1, 2, 3], p=piBB)
                actions = {'A': pAa, 'B': pBa}
                new_state, rewards, goal = world.move(actions)
                playerARewards += rewards["A"]
                playerBRewards += rewards["B"]
                curState = new_state

                if int(rewards["A"]) == 100:
                    playerAWins += 1
                if int(rewards["B"]) == 100:
                    playerBWins += 1
        # world.plot_grid()

        print("Player A Wins: ", playerAWins, "(", (float(playerAWins) / float(playerAWins + playerBWins)) * 100.0,
              "%)")
        print("Player B Wins: ", playerBWins, "(", (float(playerBWins) / float(playerAWins + playerBWins)) * 100.0,
              "%)")
        
        
        
    def play_random(self, iterations=0, log=-1):
        playerARewards = 0.0
        playerBRewards = 0.0
        playerAWins = 0
        playerBWins = 0
        RANDOM_GAMES = 500
        GameMoves = []
        for iters in range(iterations):
            # print("")
            # print("Game Number: ", iters)
            if (iters + 1) % log == 0:
                print("Random game iteration: ", iters + 1)

            validPos = [1, 2, 5, 6]
            ballStart = np.random.uniform(0, 1)
            AhasBall = True if ballStart < 0.5 else False
            BhasBall = False if ballStart < 0.5 else True
            playerAStart = validPos.pop(np.random.randint(0, 4))
            playerBStart = validPos.pop(np.random.randint(0, 3))

            pAx, pAy = self.stateToXY(playerAStart)
            pBx, pBy = self.stateToXY(playerBStart)

            player_a = Player(x=pAx, y=pAy, has_ball=AhasBall, p_id='A')
            player_b = Player(x=pBx, y=pBy, has_ball=BhasBall, p_id='B')

            world = World()
            world.set_world_size(x=self.cols, y=self.rows)
            world.place_player(player_a, player_id='A')
            world.place_player(player_b, player_id='B')
            world.set_goals(100, 0, 'A')
            world.set_goals(100, 3, 'B')

            curState = world.map_player_state()



            gms = 0

            goal = False
            while not goal:

                # 0 1 2 3
                # 4 5 6 7

                gms += 1
                piAA = [self.pi_sums[0][curState][0], self.pi_sums[0][curState][1], self.pi_sums[0][curState][2],
                        self.pi_sums[0][curState][3]]  # ,pi_sumsA[curState][4]]
                piBB = [self.pi_sums[1][curState][0], self.pi_sums[1][curState][1], self.pi_sums[1][curState][2],
                        self.pi_sums[1][curState][3]]  # ,pi_sumsB[curState][4]]
                # pAa = np.random.choice([0,1,2,3,4], p=piAA)
                # pBa = np.random.choice([0,1,2,3,4], p=piBB)
                pAa = np.random.choice([0, 1, 2, 3], p=piAA)
                pBa = np.random.choice([0, 1, 2, 3], p=piBB)
                actions = {'A': pAa, 'B': pBa}
                new_state, rewards, goal = world.move(actions)
                playerARewards += rewards["A"]
                playerBRewards += rewards["B"]
                curState = new_state
                if int(rewards["A"]) == 100:
                    playerAWins += 1
                if int(rewards["B"]) == 100:
                    playerBWins += 1
                # world.plot_grid()
                # print_status(goal, new_state, rewards, total_states)

            #GameMoves.append(gms)
        # print("Final A: ", playerARewards)
        # print("Final B: ", playerBRewards)
        # print("Player A WINS: ", playerAWins)
        # print("Player B WINS: ", playerBWins)
        print("Player A Wins: ", playerAWins, "(", (float(playerAWins) / float(playerAWins + playerBWins)) * 100.0,
              "%)")
        print("Player B Wins: ", playerBWins, "(", (float(playerBWins) / float(playerAWins + playerBWins)) * 100.0,
              "%)")
        
        
        
        
        
        
        
        
        
        
    

    def create_state_comb(self, p_a_states, p_b_states):
        """ Creates a dictionary that represents the state space possible combinations.

        Args:
            p_a_states (list): List with the numerical state labels for player A
            p_b_states (list): List with the numerical state labels for player B

        Returns:
            dict: Dictionary with the state space representation. Each element is labeled using the
                format [XYZ] where:
                    - X: shows who has the ball, either A or B.
                    - Y: state where player A is.
                    - Z: state where player B is.

                The key values hold a numeric value using the counter id_q.

        """

        states = {}
        ball_pos = ['A', 'B']
        id_q = 0

        for b in ball_pos:

            for p_a in p_a_states:

                for p_b in p_b_states:

                    if p_a != p_b:
                        states[b + str(p_a) + str(p_b)] = id_q
                        id_q += 1

        return states

class Player:

    def __init__(self, x, y, has_ball, p_id=None):
        self.p_id = p_id
        self.x = x
        self.y = y
        self.has_ball = has_ball

    def update_state(self, x, y, has_ball):
        self.x = x
        self.y = y
        self.has_ball = has_ball

    def update_x(self, x):
        self.x = x

    def update_y(self, y):
        self.y = y

    def update_ball_pos(self, has_ball):
        self.has_ball = has_ball

    def __str__(self):
        if self.x == 1 and self.y == 0:
            return "0"
        elif self.x == 1 and self.y == 1:
            return "2"
        elif self.x == 2 and self.y == 0:
            return "1"
        elif self.x == 2 and self.y == 1:
            return "3"
        else:
            return None

    def getCell(self, x, y):
        """Return state ID based on x,y
        """
        # 0 1 2 3
        # 4 5 6 7
        if x == 0 and y == 0:
            return 0
        elif x == 1 and y == 0:
            return 1
        elif x == 2 and y == 0:
            return 2
        elif x == 3 and y == 0:
            return 3
        elif x == 0 and y == 1:
            return 4
        elif x == 1 and y == 1:
            return 5
        elif x == 2 and y == 1:
            return 6
        elif x == 3 and y == 1:
            return 7
        else:
            print("Error: Invalid x,y position: ", x, ", ", y)

class World:
    """ Soccer environment simulator class. """

    def __init__(self):
        """ Method that initializes the World class with class variables to be used in the class methods. """

        self.cols = None
        self.rows = None
        self.goal_r = {}
        self.players = {}
        self.goals = {}
        self.grid = []
        self.actions = ['N', 'S', 'E', 'W', 'ST']
        self.commentator = False

    def init_grid(self):
        """ Initializes grid for plotting.

        Returns
        -------
            list: List of strings with the data to plot.

        """

        grid = [['**'] * (self.cols + 2)]

        for i in range(self.rows):
            grid.append(['**', 'gB'] + ['  '] * (self.cols - 2) + ['gA', '**'])

        grid.append(['**'] * (self.cols + 2))

        return grid

    def set_commentator_on(self):
        self.commentator = True

    def set_world_size(self, x, y):
        self.cols = x
        self.rows = y

    def set_goals(self, goal_r, goal_col, player_id):
        """ Method that sets the goal configurations.

        Args:
            goal_r (int): Reward value assigned in the event of a goal.
            goal_col (int): Column representing the goal area.
            player_id (str): Player id that will receive this reward.

        """
        self.goal_r[player_id] = goal_r
        self.goals[player_id] = goal_col

    def place_player(self, player, player_id):
        self.players[player_id] = copy.copy(player)

    def get_state_id(self, player):
        return str(player.y * self.cols + player.x)

    def map_player_state(self):
        """ Method that maps the players current state to the key used in a Q table.

        Returns
        -------
            str: Q table dict key that shows the current world state.

        """

        player_a = self.players['A']
        player_b = self.players['B']

        key = player_a.p_id if player_a.has_ball else player_b.p_id

        key += self.get_state_id(player_a)
        key += self.get_state_id(player_b)

        return key

    def get_players_states(self):
        return [self.players[p] for p in self.players]

    def plot_grid(self):
        """ Method that prints the current world state. """

        self.grid = self.init_grid()

        if len(self.players.keys()) > 0:

            for p_id in self.players:
                player = self.players[p_id]

                if player.has_ball:
                    cell = p_id + '+'

                else:
                    cell = p_id + ' '

                self.grid[player.y + 1][player.x + 1] = cell

        for r in self.grid:
            print(' | '.join(r))

        print('')

    def check_collision(self, new_pos, moving_player, other_players):
        """ Method that verifies if there is a collision between two players.

        Parameters
        ----------
            new_pos (Player): Player class instance that represents the new active player's position.
            moving_player (Player): Player class instance of the player that is moving.
            other_players (list): List of player class instances representing the other players in the game.

        Returns
        -------
            bool: True if a collision occurred, False otherwise.

        """

        collision = False

        for op_id in other_players:
            other_p = self.players[op_id]

            if new_pos.x == other_p.x and new_pos.y == other_p.y:

                if self.commentator:
                    print('{} collided with {}'.format(moving_player.p_id, other_p.p_id))

                collision = True

                if new_pos.has_ball:
                    other_p.update_ball_pos(True)
                    moving_player.update_ball_pos(False)

                    if self.commentator:
                        print("{} steals from {}".format(other_p.p_id, moving_player.p_id))

        return collision

    def check_goal(self, moving_player, player_order):
        """ Method that verifies if a goal has been scored.

        Parameters
        ----------
            moving_player (Player): Player class instance
            player_order (list): List with the order in which each agent moves

        Returns
        -------
            tuple : 2-element tuple containing
                r (int): reward obtained from scoring a goal
                goal (bool): flag that indicates a goal has been scored

        """

        goal = False
        other_players = set(player_order) - set(moving_player.p_id)
        r = {k: 0 for k in player_order}

        if moving_player.x == self.goals[moving_player.p_id] and moving_player.has_ball:

            if self.commentator:
                print("{} scored a goal!!".format(moving_player.p_id))

            goal = True
            r[moving_player.p_id] = self.goal_r[moving_player.p_id]

            for op_id in other_players:
                r[op_id] = -self.goal_r[moving_player.p_id]

        else:
            other_goal = {op_id: moving_player.x == self.goals[op_id] and moving_player.has_ball
                          for op_id in player_order}

            if sum(other_goal.values()) > 0:

                if self.commentator:
                    print("{} scored an own goal!!".format(moving_player.p_id))

                goal = True

                for k in other_goal.keys():

                    if other_goal[k]:
                        r[k] = self.goal_r[k]

                    else:
                        r[k] = -self.goal_r[k]

        return r, goal

    def move(self, a):
        """ Method that simulates a single step move command.

        Parameters
        ----------
            a (dict): actions for both players identified by the player id

        Returns
        -------
            tuple : Three-element tuple containing
                map_player_state (str): Players state after the move operation
                r (dict): rewards for each player. i.e. {A: r_a, B: r_b}
                goal (bool): flag that shows if a goal has been scored

        """

        #print(a)

        r = None
        goal = False

        player_order = list(a.keys())
        np.random.shuffle(player_order)
        new_pos = Player(0, 0, False)

        for p_id in player_order:
            moving_player = self.players[p_id]
            other_players = set(player_order) - set(p_id)
            new_pos.update_state(moving_player.x, moving_player.y, moving_player.has_ball)
            action = self.actions[a[p_id]]

            if action == 'N' and new_pos.y != 0:
                new_pos.update_y(new_pos.y - 1)

            elif action == 'E' and new_pos.x < self.cols - 1:
                new_pos.update_x(new_pos.x + 1)

            elif action == "W" and new_pos.x > 0:
                new_pos.update_x(new_pos.x - 1)

            elif action == 'S' and new_pos.y != self.rows - 1:
                new_pos.update_y(new_pos.y + 1)

            collision = self.check_collision(new_pos, moving_player, other_players)

            if not collision:
                moving_player.update_state(new_pos.x, new_pos.y, new_pos.has_ball)

            r, goal = self.check_goal(moving_player, player_order)

            if goal:
                break

            r, goal = self.check_goal(self.players[list(other_players)[0]], player_order)

            if goal:
                break

        if self.commentator:
            print('Player Order: {}'.format(player_order))
            print('Actions: {}'.format(a))
            print('A location: ({}, {})'.format(self.players['A'].x, self.players['A'].y))
            print('B location: ({}, {})'.format(self.players['B'].x, self.players['B'].y))
            print("")

        return self.map_player_state(), r, goal

    def getCell(self, player):
        if player == "A":
            xx = self.players['A'].x
            yy = self.players['A'].y
        else:
            xx = self.players['B'].x
            yy = self.players['B'].y

        if xx == 0 and yy == 0:
            return 0
        elif xx == 1 and yy == 0:
            return 1
        elif xx == 2 and yy == 0:
            return 2
        elif xx == 3 and yy == 0:
            return 3
        elif xx == 0 and yy == 1:
            return 4
        elif xx == 1 and yy == 1:
            return 5
        elif xx == 2 and yy == 1:
            return 6
        elif xx == 3 and yy == 1:
            return 7


