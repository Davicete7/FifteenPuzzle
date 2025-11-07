"""
Contains the PuzzleState class representing a state in the N-Puzzle problem.

Information:
- Each state encapsulates the board configuration, position of the empty tile (0),
- and methods for generating successors, checking goal state, and tracing solution path.

Author: 
    David Sanchez --> 905590@edu.p.lodz.pl
    Alfonso Muñoz --> 905580@edu.p.lodz.pl

"""

# Imports
import random

# Constants

# Class Definition
class PuzzleState:
    """
    Represents a state of the N-Puzzle problem (e.g., Fifteen Puzzle).

    Attributes (camelCase for variables):
        board (list[list[int]]): The current configuration of the puzzle board.
        rows (int): Number of rows (R).
        cols (int): Number of columns (C).
        zeroPos (tuple[int, int]): Position (row, col) of the empty space (0).
        parent (PuzzleState | None): The previous state from which this state was reached.
        action (str | None): The action ('L', 'R', 'U', 'D') taken to reach this state.
        gCost (int): Cost from the initial state (depth of the node).
        hCost (int): Estimated cost to the goal state (heuristic value).
        fCost (int): Total estimated cost (fCost = gCost + hCost).
        lostFCost (int): The lowest fCost of a *pruned* descendant from this node. 
                         Used by SMA* to backtrack. Initialized to hCost.
    """

    def __init__(self, board, parent=None, action=None, gCost=0, hCost=0):
        """
        Initializes a new PuzzleState.
        """
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.zeroPos = self._findZeroPos()
        self.parent = parent
        self.action = action
        self.gCost = gCost
        self.hCost = hCost
        self.fCost = self.gCost + self.hCost
        # lostFCost is crucial for SMA*. It stores the fCost of the 
        # best (lowest f) forgotten descendant. Initialize with hCost
        # as a baseline if no children have been pruned yet.
        self.lostFCost = self.hCost 
        
    def _findZeroPos(self):
        """Internal helper to find the position of the empty tile (0)."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 0:
                    return (r, c)
        return (0, 0) # Should not happen

    def isGoal(self, goalState):
        """Checks if the current state is the goal state."""
        return self.board == goalState.board
    
    # ------------------ Hashing and Comparison Methods ------------------

    def __hash__(self):
        """Allows the object to be used in sets and as dictionary keys."""
        # Convert the board to a tuple of tuples for hashing
        return hash(tuple(map(tuple, self.board)))

    def __eq__(self, other):
        """Defines the equality check (==) between two PuzzleState objects."""
        if not isinstance(other, PuzzleState):
            return NotImplemented
        return self.board == other.board
    
    def __lt__(self, other):
        """
        Defines the less than check (<) for the PriorityQueue.
        SMA* uses fCost for ordering, ties are broken by gCost (depth).
        """
        if self.fCost != other.fCost:
            return self.fCost < other.fCost
        # Tie-breaker: prefer shallower nodes (smaller gCost)
        return self.gCost < other.gCost

    def __repr__(self):
        """String representation of the object for debugging."""
        return f"State(g={self.gCost}, h={self.hCost}, f={self.fCost}, lost_f={self.lostFCost}, action={self.action})"
    
    def printBoard(self):
        """
        Imprime la configuración del tablero de forma legible.
        """
        for row in self.board:
            # Usamos str.center(3) para alinear los números de hasta 2 dígitos.
            print(" ".join(str(tile).center(3) for tile in row))

    # ------------------------ Successor Generation ------------------------

    def getSuccessors(self, order='DULR'):
        """
        Generates successor states based on a specified movement order.
        
        Args:
            order (str): String defining the order of movements (e.g., 'DULR').
        
        Returns:
            list[PuzzleState]: A list of valid successor states.
        """
        successors = []
        r, c = self.zeroPos # Current position of the empty space (0)
        
        # Directions mapping: action -> (dr, dc)
        # Note: 'L' moves the *empty space* Left, which is the same as the *piece* moving Right.
        # However, by convention, the action is named after the piece that *moves into* the empty space.
        # Or, more simply, it's named by the direction the *empty space* moves.
        # 'L': empty moves Left (piece comes from Right).
        # 'R': empty moves Right (piece comes from Left).
        # 'U': empty moves Up (piece comes from Down).
        # 'D': empty moves Down (piece comes from Up).
        
        DIRECTIONS = {
            'L': (0, -1),   # Left
            'R': (0, 1),    # Right
            'U': (-1, 0),   # Up
            'D': (1, 0)     # Down
        }
        
        # Check for random order request
        if order and order[0] == 'R':
            order_list = list(DIRECTIONS.keys())
            random.shuffle(order_list)
            order = "".join(order_list)
        
        # Iterate over the specified order
        for action in order:
            dr, dc = DIRECTIONS[action]
            newR, newC = r + dr, c + c # This was a bug: should be c + dc

            # CORRECTED:
            newR, newC = r + dr, c + dc 

            # Check if the new zero position (newR, newC) is within bounds
            if 0 <= newR < self.rows and 0 <= newC < self.cols:
                
                # We need to swap the tile at (r, c) (which is 0) with the tile at (newR, newC)
                
                # 1. Create a deep copy of the board to modify
                newBoard = [row[:] for row in self.board]
                
                # 2. Swap the zero (at r, c) with the tile at (newR, newC)
                # The piece at (newR, newC) moves into the old zero spot (r, c)
                newBoard[r][c] = newBoard[newR][newC] 
                # The zero moves to the new spot
                newBoard[newR][c] = 0             # This was a bug: should be newBoard[newR][newC] = 0
                
                # CORRECTED Swap:
                # 1. Create a deep copy of the board to modify
                newBoard = [row[:] for row in self.board]
                
                # 2. Swap the tile at (newR, newC) with the zero at (r, c)
                newBoard[r][c] = newBoard[newR][newC] # Tile at newR,newC moves to r,c
                newBoard[newR][newC] = 0             # Zero moves to newR,newC

                # 3. Create the new successor state
                # hCost is set to 0 by default, informed algorithms will calculate it later.
                newState = PuzzleState(
                    board=newBoard, 
                    parent=self, 
                    action=action, 
                    gCost=self.gCost + 1,
                    hCost=0 # Initial hCost
                )
                
                # If the successor is the same as the parent's parent, skip it (avoid immediate reverse move)
                if self.parent is not None and newState == self.parent:
                    continue

                successors.append(newState)
                
        return successors


    def getSolutionPath(self):
        """
        Traces back the parent pointers to reconstruct the solution path.
        Returns a string of actions ('L', 'R', 'U', 'D') that lead to the goal.
        """
        path = []
        currentState = self
        while currentState.parent is not None:
            path.append(currentState.action)
            currentState = currentState.parent
        
        # Path is found in reverse order, so reverse it back
        return "".join(path[::-1])
    
    def getPathFromRoot(self):
        """Helper for SMA* to reconstruct the path of a node to find its root ancestor."""
        path = [self]
        current = self
        while current.parent:
            path.append(current.parent)
            current = current.parent
        return path[::-1] # Returns path from root to self