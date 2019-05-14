from KuhnPoker.misc import *
import random


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
        #self.reach_prob =


    # Simulate playing an action
    # Returns the child node of the given action
    def play(self, action):
        return self.children[action]

    def is_chance(self):
        return self.next_player == CHANCE   # Check if chance node

    #@property
    def inf_set(self):
        raise NotImplementedError("Error: inf_set() not implemented.")


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
        self._chance_prob = 1. / len(self.children)
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

        # info set is CARD + history
        public_card = self.cards[0] if self.next_player == player0 else self.cards[1]
        self._information_set = ".{0}.{1}".format(public_card, ".".join(self.actions_history))

        totState.append(self)
        if self._information_set not in infosets:
            infosets.append(self._information_set)
            totInfoStates.append(self)

    # legal actions depending on what actions have been taken
    def __get_actions_in_next_round(self, a):
        if len(self.actions_history) == 0 and a == BET:         #first move BET => FOLD or CALL
            return [FOLD, CALL]
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