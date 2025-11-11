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
MEMORY_LIMIT = 100000000000

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
            print("Warning: Time limit exceeded (15s) for BFS.")
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
    Robust DFS:
      - 'seen' marks ON PUSH to avoid duplicates in the frontier
      - Controlled randomness WITHOUT activating the internal random of getSuccessors
      - Respects priority with LIFO (push in reverse)
    """
    import time, random

    start = time.time()

    # Normalize the input order
    order = (order or "DULR").upper()
    random_mode = order.startswith("R")
    # Base deterministic order that we will ALWAYS pass to getSuccessors (never 'R')
    base_letters = [c for c in (order if not random_mode else "DULR") if c in "DULR"]
    if not base_letters:
        base_letters = list("DULR")

    # Utility: immutable key of the board (independent of __eq__/__hash__)
    def key_of(state):
        return tuple(tuple(row) for row in state.board)

    stack = [initialState]
    seen = { key_of(initialState) }   # mark on PUSH
    nodesExpanded = 0
    maxFringeSize = 1

    while stack:
        if time.time() - start > timeLimit:
            print("Warning: Time limit exceeded (15s) for DFS.")
            return None

        current = stack.pop()

        # Goal check?
        if current.isGoal(goalState):
            return current.getSolutionPath(), nodesExpanded, maxFringeSize

        # Expansion
        nodesExpanded += 1

        # Build local order (for this expansion)
        if random_mode:
            letters = base_letters[:]   # ['D','U','L','R']
            random.shuffle(letters)
            # VERY IMPORTANT: prevent the first character from being 'R'
            if letters[0] == 'R':
                # rotate one position to ensure it does NOT start with 'R'
                letters = letters[1:] + letters[:1]
            local_order = ''.join(letters)
        else:
            local_order = ''.join(base_letters)

        # Never pass 'R' to getSuccessors; always a permutation of D/U/L/R
        successors = current.getSuccessors(order=local_order)

        # Push in REVERSE so that the first of local_order is the next to come out
        for child in reversed(successors):
            k = key_of(child)
            if k not in seen:
                seen.add(k)          # mark ON PUSH
                stack.append(child)

        if len(stack) > maxFringeSize:
            maxFringeSize = len(stack)

    # Not found (should not occur with depth 3 input)
    return None


# ---------------------------- 3. Iterative Deepening Depth-First Search (IDFS) ----------------------------
def depthLimitedSearch(node, goalState, order, limit, onPath, nodesExpanded):
    """
    DLS (canonical version): returns
      - ("solution", nodesExpanded) if it finds the goal,
      - "cutoff" if it exhausted the depth limit in some branch,
      - None if exhaustive and there is no solution at this depth.
    """
    # 1) Goal check?
    if node.isGoal(goalState):
        return node.getSolutionPath(), nodesExpanded[0]

    # 2) Out of depth budget?
    if limit == 0:
        return "cutoff"

    cutoff_occurred = False

    # 3) Expand children in deterministic order (no randomness in IDFS)
    for child in node.getSuccessors(order=order):
        if child in onPath:
            continue  # avoid loops in the current branch

        onPath.add(child)
        nodesExpanded[0] += 1

        result = depthLimitedSearch(child, goalState, order, limit - 1, onPath, nodesExpanded)

        onPath.remove(child)

        if result == "cutoff":
            cutoff_occurred = True
        elif result is not None:
            # Found solution in a descendant
            return result

    # 4) Result propagation
    return "cutoff" if cutoff_occurred else None


def iterativeDeepeningDFS(initialState, goalState, order, timeLimit=60):
    """
    IDFS: increments limit 0,1,2,... until finding a solution or running out of time.
    """
    import time
    start = time.time()

    # Normalize order and avoid randomness in IDFS
    order = (order or "DULR").upper()
    if order in ("R", "RAND", "RANDOM"):
        order = "DULR"  # deterministic for reproducibility

    totalNodesExpanded = 0
    maxFringeSize = 1

    # A reasonable upper limit; for 3x3 << 50 is enough
    MAX_DEPTH = 50

    for depth in range(0, MAX_DEPTH + 1):
        if time.time() - start > timeLimit:
            return None

        onPath = {initialState}    # only for the current iteration
        nodesExpandedIter = [0]

        result = depthLimitedSearch(initialState, goalState, order, depth, onPath, nodesExpandedIter)

        totalNodesExpanded += nodesExpandedIter[0]
        # In IDFS the approximate "max frontier" = current depth + 1
        if depth + 1 > maxFringeSize:
            maxFringeSize = depth + 1

        if result is None:
            # Exhaustive search at this limit without cutoffs -> no solution
            return None
        if result != "cutoff":
            # Solution found!
            solutionPath, _ = result
            return solutionPath, totalNodesExpanded, maxFringeSize

    # Exhausted MAX_DEPTH without solution
    return None


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
            print("Warning: Time limit exceeded (15s) for Best-First Search.")
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
            print("Warning: Time limit exceeded (15s) for A* Search.")
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
            sys.stderr.write("Warning: Time limit exceeded (15s) for SMA*.\n")
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