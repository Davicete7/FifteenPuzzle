"""
Utility functions for generating the goal board configuration
for the N-Puzzle problem.

Author:
    David Sanchez --> 905590@edu.p.lodz.pl
    Alfonso Muñoz --> 905580@edu.p.lodz.pl
"""

# Imports

# Constants


# Functions
def getGoalBoard(rows, cols):
    """
    Generates the ordered target state board for the R x C puzzle.
    The goal is ordered (1, 2, 3, ...), with '0' (empty space) 
    at the bottom-right corner (rows-1, cols-1).
    
    Args:
        rows (int): Number of rows (R).
        cols (int): Number of columns (C).

    Returns:
        list[list[int]]: The goal board configuration.
    """
    goalBoard = []
    tileValue = 1
    totalTiles = rows * cols # Total number of cells (e.g., 16 for 4x4)
    
    for r in range(rows):
        currentBoardRow = []
        for c in range(cols):
            
            # If tileValue equals the total number of tiles, we are at the last cell.
            # This is where the empty space (0) should be placed.
            if tileValue == totalTiles:
                currentBoardRow.append(0) 
            else:
                currentBoardRow.append(tileValue)
            
            tileValue += 1
            
        goalBoard.append(currentBoardRow)
        
    return goalBoard

def heuristicZero(currentState, goalState):
    """
    Heuristic h(x) = 0. 
    Used to implement Uniform Cost Search (A* with h=0).
    
    Args:
        currentState (PuzzleState): The current puzzle state.
        goalState (PuzzleState): The target goal state.
        
    Returns:
        int: Always 0.
    """
    return 0

def heuristicMisplacedTiles(currentState, goalState):
    """
    Misplaced Tiles Heuristic.
    Counts the number of tiles that are not in their goal position.
    (Excludes the empty space (0)).
    """
    # Obtain R and C from the current board for greater robustness
    R = len(currentState.board)
    C = len(currentState.board[0])
    misplacedCount = 0

    for r in range(R):
        for c in range(C):
            piece = currentState.board[r][c]
            # Exclude the empty space (0) from the count
            if piece != 0 and piece != goalState.board[r][c]:
                misplacedCount += 1
                
    return misplacedCount

def heuristicManhattanDistance(currentState, goalState):
    """
    Manhattan Distance Heuristic.
    Sums the Manhattan distance (horizontal + vertical) for each tile
    from its current position to its goal position.
    (Excludes the empty space (0)).
    """
    # Obtain R and C from the current board for greater robustness
    R = len(currentState.board)
    C = len(currentState.board[0])
    totalDistance = 0
    
    # Dictionary to store the goal position of each tile (value -> (r, c))
    goalPositions = {}
    for r_goal in range(R):
        for c_goal in range(C):
            piece = goalState.board[r_goal][c_goal]
            goalPositions[piece] = (r_goal, c_goal)
    
    # Calculate the total distance
    for r_current in range(R):
        for c_current in range(C):
            piece = currentState.board[r_current][c_current]
            
            # Ignore the empty space (0)
            if piece != 0:
                # Ensure the piece is in goalPositions (it should be)
                if piece in goalPositions:
                    r_goal, c_goal = goalPositions[piece]
                    
                    # Manhattan Distance = |r_actual - r_goal| + |c_actual - c_goal|
                    distance = abs(r_current - r_goal) + abs(c_current - c_goal)
                    totalDistance += distance
                
    return totalDistance


# Heuristic mapping for easy access
HEURISTIC_MAP = {
    0: heuristicZero,
    1: heuristicMisplacedTiles,
    2: heuristicManhattanDistance,
}