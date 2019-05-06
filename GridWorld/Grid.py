



class Grid(object):
    """Creates GridWorld


    """

    def __init__(self):

        # Action direction constants
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        # Dimensions
        self.rows = 4
        self.cols = 12

        self.livingCost = -1

        self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 126]

        self.rewards = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 126]


    def getMove(self, s, action):
        """

        :param action:
        :return: Next State
        """
        appActX, appActY = self.applyAction(s, action)
        validMove = self._isValidMove(appActX, appActY)
        if validMove:
            return self.XYToState(appActX, appActY)
        else:
            return s

    def applyAction(self, s, action):
        x = s // self.cols
        y = s % self.cols

        if action == self.UP:
            return (x - 1, y)
        elif action == self.RIGHT:
            return (x, y + 1)
        elif action == self.DOWN:
            return (x + 1, y)
        elif action == self.LEFT:
            return (x, y - 1)

    def _isValidMove(self, x, y):
        if x < 0 or x >= self.rows: return False
        if y < 0 or y >= self.cols: return False
        return True

    def XYToState(self, x, y):
        return x * self.cols + y