
from itertools import permutations

# Total possible card pairs
# - total of 6 possible scenarios for a game
JQ = 'JQ'
JK = 'JK'
QJ = 'QJ'
QK = 'QK'
KJ = 'KJ'
KQ = 'KQ'

J = 'J'
Q = 'Q'
K = 'K'



# list with all 6 possible game configurations
deck = [JQ, JK, QJ, QK, KJ,KQ]


# One chance node - at the top of the tree
# - signifying the different deals of cards
CHANCE = "CHANCE"

# Valid moves in game (not in every state)

CHECK = "CHECK"
CALL = "CALL"
FOLD = "FOLD"
BET = "BET"

CONTINUE = "CONTINUE"
PASS = "PASS"


#leduc deck is AH KH AS KS AC KC
AH = 'AH' #A
KH = 'KH' #B
QH = 'QH' #C
AS = 'AS' #D
KS = 'KS' #E
QS = 'QS' #F
A = 'A'
B = 'B'
C = 'C'
D = 'D'
E = 'E'
F = 'F'

numToCard = {}
numToCard[A] = AH
numToCard[B] = KH
numToCard[C] = QH
numToCard[D] = AS
numToCard[E] = KS
numToCard[F] = QS


#player IDs - 1 and -1 useful for some math operations throughout
player0 = 1
player1 = -1

# dict for analyzing if player0 (index 0) won or lost
winMap = {}
winMap[QK] = -1
winMap[JK] = -1
winMap[JQ] = -1
winMap[KQ] = 1
winMap[KJ] = 1
winMap[QJ] = 1


# Initialize all strategies
# - Map all infosets to another dict where each action is initialized to be uniform
# - this method is called once at start where node = root
# RETURNS: dict[node.information_set][action] => strategy probability (uniform)
def init_strategy(node, output=None):

    def init_strategy_recursive(node):

        if node is None:
            return

        node_inf_set = node.inf_set()
        output[node_inf_set] = {action: 1.0 / len(node.actions) for action in node.actions}
        for k in node.children:
            init_strategy_recursive(node.children[k])

    output = dict()
    init_strategy_recursive(node)
    return output


# Initialize all of the maps in each node
# - called before init_strategy() above
# - RETURNS: dict[node.information_set][action] => strategy probability (0 init)
def init_empty_node_maps(node, output = None):

    def init_empty_node_maps_recursive(node):

        if node is None:
            return

        node_inf_set = node.inf_set()

        output[node_inf_set] = {action: 0. for action in node.actions}
        for k in node.children:
            init_empty_node_maps_recursive(node.children[k])

    output = dict()
    init_empty_node_maps_recursive(node)
    return output












