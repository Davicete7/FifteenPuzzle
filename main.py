"""
Task 3 --> Solving problems by searching - Fifteen puzzle

Authors: 
    David Sánchez -> 905590@edu.p.lodz.pl
    Alfonso Muñoz -> 905580@edu.p.lodz.pl
"""

# Imports
import sys
import time
from puzzleState import PuzzleState
from searchUtils import getGoalBoard, HEURISTIC_MAP
from searchAlgorithms import (
    breadthFirstSearch,
    depthFirstSearch,
    iterativeDeepeningDFS,
    bestFirstSearch,
    aStarSearch,
    smaStarSearch
)

# Constants
INPUT_FILE_PATH = (r"input.txt")

# Global map for easier selection of search functions
SEARCH_FUNCTIONS = {
    '1': ('BFS', breadthFirstSearch, 'order'),
    '2': ('DFS', depthFirstSearch, 'order'),
    '3': ('IDFS', iterativeDeepeningDFS, 'order'),
    '4': ('Best First', bestFirstSearch, 'heuristic'),
    '5': ('A*', aStarSearch, 'heuristic'),
    '6': ('SMA*', smaStarSearch, 'heuristic'),
}


# Functions
def visualizeSolution(initialState, solutionPath):
    """
    Toma el estado inicial y la ruta de solución para visualizar el proceso paso a paso.
    """
    if not solutionPath:
        print("\n--- There is no solution to visualize. ---")
        return

    print("\n" + "="*50)
    print("🎬 START OF SOLUTION VISUALIZATION 🎬")
    print("="*50)
    
    currentState = initialState
    
    # Muestra el estado inicial
    print("\n--- Step 0: Initial State ---")
    currentState.printBoard()
    print("-" * 20)
    
    # Recorre cada acción de la ruta
    for i, action in enumerate(solutionPath):
        # Generamos el siguiente estado. 
        # NOTA: getSuccessors genera una lista, pero solo un sucesor corresponde a 'action'.
        
        # Obtenemos el único sucesor que corresponde a la acción.
        # Esto asume que PuzzleState.getSuccessors puede tomar un orden de un solo movimiento.
        # Si no, necesitamos obtener todos y buscar el que coincide con 'action'.
        
        # Una forma robusta:
        successors = currentState.getSuccessors(order=action)
        
        # Si 'successors' contiene estados que resultan del movimiento 'action'
        if successors and successors[0].action == action:
            nextState = successors[0]
            
            # Reemplazamos el estado actual con el siguiente para la siguiente iteración
            currentState = nextState

            print(f"\n--- Step {i+1}: Move {action} ---")
            currentState.printBoard()
            print("-" * 20)
        else:
            # Esto no debería ocurrir si el algoritmo de búsqueda funcionó correctamente.
            print(f"ERROR: The move '{action}' failed at step {i+1}.")
            break

    print("\n" + "="*50)
    print("✅ END OF VISUALIZATION. Final State Achieved.")
    print("="*50)

def readInput():
    """
    Reads the puzzle configuration from the specified file.
    
    Returns:
        tuple[list[list[int]] | None, int | None, int | None]: 
        The initial board configuration, row count (R), and column count (C).
    """
    try:
        with open(INPUT_FILE_PATH, 'r') as inputFile:
            
            # 1. Read R (rows) and C (columns) from the first line
            rcLine = inputFile.readline().strip().split()
            if not rcLine:
                sys.stderr.write(f"Error: File '{INPUT_FILE_PATH}' is empty or missing dimensions line.\n")
                return None, None, None
                
            rowCount, colCount = map(int, rcLine)
            
            initialBoard = []
            
            # 2. Read R subsequent lines
            for r in range(rowCount):
                line = inputFile.readline().strip()
                if not line:
                    raise EOFError(f"Incomplete board data in '{INPUT_FILE_PATH}'. Expected {rowCount} rows, found {r}.")
                    
                row = list(map(int, line.split()))
                
                if len(row) != colCount:
                    raise ValueError(f"Row {r+1} length ({len(row)}) in '{INPUT_FILE_PATH}' does not match column count ({colCount}).")
                    
                initialBoard.append(row)

            # 3. Final Validation: Ensure the empty space (0) is present
            if not any(0 in row for row in initialBoard):
                 raise ValueError(f"The puzzle board in '{INPUT_FILE_PATH}' must contain the value 0 (empty space).")

            return initialBoard, rowCount, colCount
            
    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file '{INPUT_FILE_PATH}' not found.\n")
        return None, None, None
    except Exception as e:
        sys.stderr.write(f"Error reading input file '{INPUT_FILE_PATH}': {e}\n")
        return None, None, None


def getSearchParameters(searchType):
    """
    Prompts the user for the necessary parameter (order or heuristic ID).
    """
    if searchType in ['BFS', 'DFS', 'IDFS']:
        param = input("Enter expansion ORDER (DULR, or R for Random):").strip().upper()
        if not all(c in 'LRUD' for c in param) and not param.startswith('R'):
             print("Invalid order. Must be a permutation of LRUD or start with R.")
             return None
        return param
    
    elif searchType in ['Best First', 'A*', 'SMA*']:
        try:
            paramId = int(input("Enter Heuristic ID (0: h=0, 1: Misplaced, 2: Manhattan): ").strip())
            if paramId not in HEURISTIC_MAP:
                print("Invalid Heuristic ID. Must be 0, 1, or 2.")
                return None
            return HEURISTIC_MAP[paramId] # Returns the function reference
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, or 2).")
            return None
    return None


def printResults(result, startTime, searchType, param):
    """
    Prints the solution path and statistics to standard output.
    """
    endTime = time.time()
    elapsedTime = endTime - startTime
    
    print("-" * 50)
    print(f"Algorithm: {searchType}")
    print(f"Parameter: {param if isinstance(param, str) else param.__name__.replace('heuristic', 'h')}")
    print(f"Execution Time: {elapsedTime:.4f} seconds")
    print("-" * 50)

    if result is None:
        print("-1") # Length of solution: -1
        print("")  # Solution string: empty
        print("Status: Puzzle could not be solved or search limit reached.")
        return

    solutionPath, nodesExpanded, maxFringeSize = result
    
    # OUTPUT REQUIRED FORMAT
    print(len(solutionPath))
    print(solutionPath)
    
    # Statistics (for technical report)
    print(f"Path Length (gCost): {len(solutionPath)}")
    print(f"Nodes Expanded: {nodesExpanded}")
    print(f"Max Fringe Size: {maxFringeSize}")
    print("-" * 50)


def printMainMenu():
    """
    Display the main menu for selecting a search algorithm.
    """
    print("\n\n\n--- MENU: Select Search Algorithm ---")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Iterative Deepening DFS (IDFS)")
    print("4. Best First Search ")
    print("5. A* Search")
    print("6. Simplified Memory-Bounded A* (SMA*)")
    print("0. Exit")


def main():
    """
    Main function to run the search algorithm selection menu.
    """
    # 1. READ INPUT AND INITIALIZE STATES ONCE
    initialBoard, rows, cols = readInput()
    if initialBoard is None:
        return # Exit on input error
        
    initialState = PuzzleState(initialBoard) 
    goalBoard = getGoalBoard(rows, cols)
    goalState = PuzzleState(goalBoard)
    
    # Map for easy lookup of search functions
    searchMap = {
        '1': breadthFirstSearch,
        '2': depthFirstSearch,
        '3': iterativeDeepeningDFS,
        '4': bestFirstSearch,
        '5': aStarSearch,
        '6': smaStarSearch
    }


    while True:
        printMainMenu()
        elec = input("\n\nSelect an option (1-6): ").strip()
        
        if elec == '0':
            print("Exit selected. Goodbye!")
            break

        if elec not in SEARCH_FUNCTIONS:
            print("Invalid option. Please try again.")
            continue
            
        searchType, searchFunc, paramType = SEARCH_FUNCTIONS[elec]
        
        # 2. Get the necessary parameter (order or heuristic function)
        param = getSearchParameters(searchType)
        
        if param is not None and searchFunc is not None:
            
            startTime = time.time()
            
            # 3. Call the search function
            # The 'param' holds the 'order' string or the 'h_func' reference
            try:
                # 3. Call the search function
                result = searchFunc(initialState, goalState, param) 
                
                # 4. Print results and statistics
                
                if result:
                    solutionPath, _, _ = result # Desempaquetamos solo la ruta para la visualización
                    
                    # Llamamos a printResults con el resultado completo y otros parámetros
                    printResults(result, startTime, searchType, param) 
                    
                    # --- NEW VISUALIZATION PART ---
                    if solutionPath: # If a solution was found
                        visualize_option = input("\nDo you want to visualize the solution step by step? (Y/N): ").strip().upper()
                        if visualize_option == 'Y':
                            # Call the visualization function
                            visualizeSolution(initialState, solutionPath)
                    # ------------------------------------
                    
                else:
                    # Llama a printResults incluso si no hay solución para imprimir el -1
                    printResults(None, startTime, searchType, param)

            except Exception as e:
                sys.stderr.write(f"An error occurred during {searchType}: {e}\n")
                
        elif searchFunc is None:
            print(f"Algorithm {searchType} is not yet implemented.")


#-------------------------------------MAIN PROGRAM-------------------------------------#

if __name__ == "__main__":
    main()