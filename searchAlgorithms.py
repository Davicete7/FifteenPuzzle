"""
    Implementation of different search algorithms for solving the N-Puzzle problem.

    Author:
        David Sanchez --> 905590@edu.p.lodz.pl
        Alfonso Muñoz --> 905580@edu.p.lodz.pl
"""

# Imports
from collections import deque
import time
import sys
from queue import PriorityQueue

# Constants
MEMORY_LIMIT = 50000

# Functions
# ---------------------------- 1. Breadth-First Search (BFS) ----------------------------
def breadthFirstSearch(initialState, goalState, order, timeLimit=15):
    """
    Performs the Breadth-First Search (BFS) algorithm.

    Args:
        initialState (PuzzleState): The starting configuration.
        goalState (PuzzleState): The target configuration.
        order (str): The order of successor expansion (e.g., 'DULR').
        timeLimit (int): Maximum time (in seconds) allowed for the search.

    Returns:
        tuple: (solutionPath, nodesExpanded, maxFringeSize) or None if unsolvable.
    """
    
    startTime = time.time()
    
    # Queue (FIFO) for the frontier (nodes to visit)
    frontier = deque([initialState])
    
    # Set for visited states (for fast O(1) lookup)
    # Stores the hash of the board configuration to avoid cycles and revisits
    visited = {initialState} 
    
    nodesExpanded = 0
    maxFringeSize = 1

    # Main Search Loop
    while frontier:
        
        # Check time limit
        if time.time() - startTime > timeLimit:
            print("Warning: Time limit exceeded (60s) for BFS.")
            return None 

        # Update max fringe size for statistics
        maxFringeSize = max(maxFringeSize, len(frontier))
        
        # Get the node with the least depth (FIFO)
        currentState = frontier.popleft()
        
        # Goal Check
        if currentState.isGoal(goalState):
            # Return solution path and statistics
            solutionPath = currentState.getSolutionPath()
            return solutionPath, nodesExpanded, maxFringeSize

        # Expansion
        nodesExpanded += 1
        
        # Generate successors using the specified order
        for successor in currentState.getSuccessors(order):
            
            # Check if the state has NOT been visited
            if successor not in visited:
                
                # Add to visited set and the frontier queue
                visited.add(successor)
                frontier.append(successor)

    # Puzzle is unsolvable
    return None


# ---------------------------- 2. Depth-First Search (DFS) ----------------------------
def depthFirstSearch(initialState, goalState, order, timeLimit=15):
    """
    Performs the Depth-First Search (DFS) algorithm.
    Uses a stack (LIFO) and a strict visited set to prevent infinite loops.

    Args:
        initialState (PuzzleState): The starting configuration.
        goalState (PuzzleState): The target configuration.
        order (str): The order of successor expansion (e.g., 'DULR').
        timeLimit (int): Maximum time (in seconds) allowed for the search.

    Returns:
        tuple: (solutionPath, nodesExpanded, maxFringeSize) or None if unsolvable.
    """
    
    startTime = time.time()
    
    # Stack (LIFO: use Python list and append/pop) for the frontier
    frontier = [initialState] 
    
    # Set of visited states to avoid cycles (CRUCIAL for general DFS on graphs/trees with cycles)
    visited = {initialState} 
    
    nodesExpanded = 0
    maxFringeSize = 1

    # Main Search Loop
    while frontier:
        
        # Check time limit
        if time.time() - startTime > timeLimit:
            print("Warning: Time limit exceeded (60s) for DFS.")
            return None 
            
        # Update max fringe size for statistics
        maxFringeSize = max(maxFringeSize, len(frontier))
        
        # Get the most recently added node (LIFO: last one in)
        currentState = frontier.pop()
        
        # Goal Check
        if currentState.isGoal(goalState):
            solutionPath = currentState.getSolutionPath()
            return solutionPath, nodesExpanded, maxFringeSize

        # Expansion
        nodesExpanded += 1
        
        # Generate successors in REVERSE order if using LIFO stack, 
        # to ensure the first specified move is explored deepest first.
        # Example: if order is 'DULR', we process 'R', then 'L', 'U', 'D' so 'D' 
        # is the last one pushed and therefore the first one popped (deepest exploration).
        # We also want to reverse the successors list itself.
        successors = currentState.getSuccessors(order)
        
        # Iterate over successors in reverse order
        for successor in reversed(successors):
            
            if successor not in visited:
                visited.add(successor)
                frontier.append(successor)

    # Puzzle is unsolvable
    return None


# ---------------------------- 3. Iterative Deepening Depth-First Search (IDFS) ----------------------------
def depthLimitedSearch(currentState, goalState, order, depthLimit, visited, nodesExpanded):
    """
    Performs Depth-Limited Search (DLS), a recursive subroutine for IDFS.

    Args:
        currentState (PuzzleState): The current node to check.
        goalState (PuzzleState): The target goal configuration.
        order (str): The order of successor expansion (used to determine child order).
        depthLimit (int): The maximum depth to explore in this iteration.
        visited (set): Set of states visited ALONG the current path (to prevent cycles).
        nodesExpanded (list[int]): Counter for expanded nodes, passed as a mutable list [0].

    Returns:
        tuple | str: (solutionPath, nodesExpanded) if goal is found, "cutoff" if the 
                     depth limit is reached, or None if no solution is found within 
                     the current limit and no cutoff occurred.
    """
    
    # 1. Check for goal state
    if currentState.isGoal(goalState):
        return currentState.getSolutionPath(), nodesExpanded[0]

    # 2. Check depth limit
    if currentState.gCost >= depthLimit:
        return "cutoff", nodesExpanded[0]

    # 3. Expansion
    nodesExpanded[0] += 1
    
    cutoffOccurred = False
    
    # Generate successors
    successors = currentState.getSuccessors(order)

    # Iterate over successors
    for successor in successors:
        
        # Prevent cycles within the current path (crucial for DFS/DLS correctness)
        if successor not in visited:
            visited.add(successor)
            
            # Recursive call to DLS
            result = depthLimitedSearch(successor, goalState, order, depthLimit, visited, nodesExpanded)
            
            # Remove from 'visited' as we backtrack (this state is no longer on the current path)
            visited.remove(successor)

            if result == "cutoff":
                cutoffOccurred = True
            elif result is not None:
                # Goal found: return the complete result
                return result

    # Return "cutoff" if any descendant returned "cutoff", otherwise None (search space exhausted at this branch)
    return "cutoff" if cutoffOccurred else None


def iterativeDeepeningDFS(initialState, goalState, order, timeLimit=15):
    """
    Performs Iterative Deepening Depth-First Search (IDFS).

    Args:
        initialState (PuzzleState): The initial configuration.
        goalState (PuzzleState): The target goal configuration.
        order (str): The order of successor expansion.
        timeLimit (int): Maximum time (in seconds) allowed for the search.

    Returns:
        tuple: (solutionPath, totalNodesExpanded, maxFringeSize) or None if unsolvable.
    """
    startTime = time.time()
    maxFringeSize = 1
    totalNodesExpanded = 0
    
    # Set a reasonable maximum depth
    maxDepth = 50 
    
    for depthLimit in range(maxDepth):
        
        if time.time() - startTime > timeLimit:
            print(f"Warning: Time limit exceeded ({timeLimit}s) for IDFS.")
            return None 

        # 'visited' is local to the DLS call, preventing cycles *within the current path*
        visited = {initialState} 
        nodesExpandedCurrentIteration = [0] 
        
        # Call DLS
        result = depthLimitedSearch(initialState, goalState, order, depthLimit, visited, nodesExpandedCurrentIteration)
        
        totalNodesExpanded += nodesExpandedCurrentIteration[0]
        maxFringeSize = max(maxFringeSize, depthLimit + 1)
        
        if result is not None and result != "cutoff":
            # Solution found!
            solutionPath, _ = result 
            return solutionPath, totalNodesExpanded, maxFringeSize

        elif result is None:
            # Search space exhausted (no cutoff)
            return None

    return None # If maxDepth is reached


# ---------------------------- 4. Best-First Search (Greedy Search) ----------------------------
def bestFirstSearch(initialState, goalState, h_func, timeLimit=15):
    """
    Performs the Best-First Search (Greedy Search) algorithm.
    Expands nodes based on the lowest heuristic cost (hCost).

    Args:
        initialState (PuzzleState): The starting configuration.
        goalState (PuzzleState): The target configuration.
        h_func (function): The heuristic function (e.g., heuristicManhattanDistance).
        timeLimit (int): Maximum time (in seconds) allowed for the search.

    Returns:
        tuple: (solutionPath, nodesExpanded, maxFringeSize) or None if unsolvable.
    """
    
    startTime = time.time()
    
    # 1. Initialize the initial state's hCost
    initialState.hCost = h_func(initialState, goalState)
    
    # PriorityQueue: stores (hCost, currentState). Lowest hCost is first.
    frontier = PriorityQueue()
    frontier.put((initialState.hCost, initialState))
    
    # Set to store visited states (using hash of board config)
    visited = {initialState} 
    
    nodesExpanded = 0
    maxFringeSize = 1

    # Main Search Loop
    while not frontier.empty():
        
        # Check time limit
        if time.time() - startTime > timeLimit:
            print("Warning: Time limit exceeded (60s) for Best-First Search.")
            return None 
            
        # Update max fringe size
        maxFringeSize = max(maxFringeSize, frontier.qsize())
        
        # Get the state with the LOWEST hCost.
        _, currentState = frontier.get()
        
        # Goal Check
        if currentState.isGoal(goalState):
            solutionPath = currentState.getSolutionPath()
            return solutionPath, nodesExpanded, maxFringeSize

        # Expansion
        nodesExpanded += 1
        
        # Generate successors (use default order 'DULR' as PriorityQueue handles priority)
        successors = currentState.getSuccessors(order='DULR') 
        
        # Iterate over successors
        for successor in successors:
            
            if successor not in visited:
                
                # 2. Calculate the heuristic cost for the new state
                successor.hCost = h_func(successor, goalState)
                
                visited.add(successor)
                # Add to frontier: (hCost, successor)
                frontier.put((successor.hCost, successor))

    # If the loop finishes without finding the goal
    return None

# ---------------------------- 5. A* Search ----------------------------
def aStarSearch(initialState, goalState, h_func, timeLimit=15):
    """
    Performs the A* Search algorithm.
    Expands nodes based on the lowest total estimated cost: f(n) = g(n) + h(n).

    Args:
        initialState (PuzzleState): The starting configuration.
        goalState (PuzzleState): The target configuration.
        h_func (function): The heuristic function (e.g., heuristicManhattanDistance).
        timeLimit (int): Maximum time (in seconds) allowed for the search.

    Returns:
        tuple: (solutionPath, nodesExpanded, maxFringeSize) or None if unsolvable.
    """
    
    startTime = time.time()
    
    # 1. Initialize the initial state's hCost and fCost
    initialState.hCost = h_func(initialState, goalState)
    # fCost = gCost + hCost (gCost is 0 for the initial state)
    initialState.fCost = initialState.gCost + initialState.hCost
    
    # PriorityQueue: stores (fCost, currentState). Lowest fCost is first.
    # We use a tuple (fCost, unique_id, currentState) to break ties reliably, 
    # as PriorityQueue requires comparable elements.
    frontier = PriorityQueue()
    unique_id = 0
    frontier.put((initialState.fCost, unique_id, initialState))
    unique_id += 1
    
    # Dictionary to store the lowest gCost found so far for each visited state.
    # This is CRUCIAL for A* optimality and graph search. Key: state, Value: gCost.
    gCostTrack = {initialState: initialState.gCost}
    
    nodesExpanded = 0
    maxFringeSize = 1

    # Main Search Loop
    while not frontier.empty():
        
        # Check time limit
        if time.time() - startTime > timeLimit:
            print("Warning: Time limit exceeded (60s) for A* Search.")
            return None 
            
        # Update max fringe size
        maxFringeSize = max(maxFringeSize, frontier.qsize())
        
        # Get the state with the LOWEST fCost. We discard the unique_id.
        _, _, currentState = frontier.get()
        
        # Critical Check for A* and PriorityQueue:
        # If we pulled a state from the queue whose gCost is already higher 
        # than a previously found path to that same state (stored in gCostTrack),
        # we ignore it because it's an outdated, suboptimal entry in the queue.
        if currentState.gCost > gCostTrack.get(currentState, float('inf')):
            continue

        # Goal Check
        if currentState.isGoal(goalState):
            solutionPath = currentState.getSolutionPath()
            return solutionPath, nodesExpanded, maxFringeSize

        # Expansion
        nodesExpanded += 1
        
        # Generate successors (use default order 'DULR' as PriorityQueue handles priority)
        successors = currentState.getSuccessors(order='DULR') 
        
        # Iterate over successors
        for successor in successors:
            
            # gCost of the successor (current gCost + 1)
            new_gCost = successor.gCost
            
            # Check if this new path to the successor is better than any previous path found
            if new_gCost < gCostTrack.get(successor, float('inf')):
                
                # 2. Update costs for the successor
                successor.hCost = h_func(successor, goalState)
                successor.fCost = new_gCost + successor.hCost # f(n) = g(n) + h(n)
                
                # 3. Update the optimal gCost for this state
                gCostTrack[successor] = new_gCost
                
                # 4. Add the successor to the frontier
                frontier.put((successor.fCost, unique_id, successor))
                unique_id += 1

    # If the loop finishes without finding the goal
    return None


# ---------------------------- 6. Simplified Memory-Bounded A* (SMA*) ----------------------------
def smaStarSearch(initialState, goalState, h_func, timeLimit=15):
    """
    Realiza la búsqueda SMA* (Simplified Memory-Bounded A*).

    SMA* limita el tamaño de la frontera (nodos a explorar). Si el límite de 
    memoria se excede, el nodo con el F-cost más alto (el peor) es desechado.
    
    Args:
        initialState (PuzzleState): La configuración inicial.
        goalState (PuzzleState): La configuración objetivo.
        h_func (function): La función heurística a utilizar.
        timeLimit (int): Tiempo máximo (en segundos) permitido para la búsqueda.

    Returns:
        tuple: (solutionPath, nodesExpanded, maxFringeSize) o None si no se resuelve.
    """
    
    startTime = time.time()
    
    # 1. Inicialización del estado inicial: Calcular H y F
    initialState.hCost = h_func(initialState, goalState)
    initialState.fCost = initialState.gCost + initialState.hCost
    
    # La frontera (PriorityQueue) almacena tuplas: (fCost, unique_id, node)
    # Usamos unique_id para desempatar nodos con el mismo fCost
    frontier = PriorityQueue()
    unique_id = 0 
    frontier.put((initialState.fCost, unique_id, initialState))
    unique_id += 1
    
    # Diccionario para rastrear el costo g más bajo encontrado para cada estado
    # Esto es crucial para la re-exploración (path re-costing) en A*
    gCostTrack = {initialState: initialState.gCost}
    
    nodesExpanded = 0
    maxFringeSize = 1
    memoryLimit = MEMORY_LIMIT 

    # Bucle Principal de Búsqueda
    while not frontier.empty():
        
        # Comprobación de límite de tiempo
        if time.time() - startTime > timeLimit:
            sys.stderr.write("Warning: Time limit exceeded (60s) for SMA*.\n")
            return None 

        # Actualizar el tamaño máximo de la frontera
        maxFringeSize = max(maxFringeSize, frontier.qsize())
        
        # Obtener el mejor nodo (menor fCost)
        _, _, currentState = frontier.get()
        
        # Ignorar si ya encontramos un camino mejor a este estado (entrada "stale" en la cola)
        if currentState.gCost > gCostTrack.get(currentState, float('inf')):
             continue

        # Comprobación de Meta
        if currentState.isGoal(goalState):
            solutionPath = currentState.getSolutionPath()
            return solutionPath, nodesExpanded, maxFringeSize

        # Expansión
        nodesExpanded += 1
        
        # Generar sucesores (usando el orden por defecto 'DULR' o un parámetro si lo has definido)
        successors = currentState.getSuccessors(order='DULR') 

        for successor in successors:
            
            new_gCost = successor.gCost
            
            # Si hemos encontrado un mejor camino al sucesor
            if new_gCost < gCostTrack.get(successor, float('inf')):
                
                # Actualizar costos
                successor.hCost = h_func(successor, goalState)
                successor.fCost = new_gCost + successor.hCost
                
                # Actualizar el rastro del costo g más bajo
                gCostTrack[successor] = new_gCost
                
                # Insertar en la frontera
                frontier.put((successor.fCost, unique_id, successor))
                unique_id += 1


        # Gestión de Memoria (Estrategia de poda SMA*)
        if frontier.qsize() > memoryLimit:
             # ESTA PARTE ES INEFICIENTE, PERO FUNCIONAL: VACIAR, BUSCAR EL MÁXIMO, Y RELLENAR
             nodes_list = []
             # 1. Vaciar la cola
             while not frontier.empty():
                 nodes_list.append(frontier.get())

             # 2. Encontrar el índice del nodo con el fCost más alto (el peor)
             worst_node_index = -1
             max_fCost = -1
             
             for i, (f, _, _) in enumerate(nodes_list):
                 if f > max_fCost:
                     max_fCost = f
                     worst_node_index = i

             # 3. Eliminar el peor nodo (La Poda)
             if worst_node_index != -1:
                 # Nota: Aquí la SMA* original propagaría max_fCost a la fCost del padre si es peor que su fCost actual.
                 nodes_list.pop(worst_node_index) 

             # 4. Reinsertar los nodos restantes en la cola de prioridad
             for item in nodes_list:
                 frontier.put(item)
                    
    # Si el bucle termina sin encontrar el objetivo
    return None