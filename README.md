🧩 N-Puzzle Solver (Fifteen Puzzle)

A robust implementation of uninformed and informed search algorithms to solve the classic logic game, the N-Puzzle, with a specific focus on the Fifteen Puzzle (4x4).

This project allows users to define an initial state and apply various Artificial Intelligence search methods to find the optimal or sub-optimal solution path, with step-by-step visualization and performance analysis.

🌟 Key Features

Implemented Search Algorithms:

Uninformed Search: Breadth-First Search (BFS), Depth-First Search (DFS), Iterative Deepening Depth-First Search (IDFS).

Informed Search: Best-First Search (Best-First Search), A* Algorithm, and SMA* (Simplified Memory-Bounded A*).

Heuristics: Support for key admissible heuristics, including Manhattan Distance and Misplaced Tiles.

Modular Structure: Code is organized into dedicated modules (puzzleState.py, searchAlgorithms.py, main.py) ensuring clear separation of state logic, search methods, and the main execution flow.

Solution Visualization: Interactive console output function that displays the solution step-by-step, recreating the sequence of board moves.

Performance Analysis: Provides detailed statistics for each execution: CPU time, path length, number of expanded nodes, and maximum fringe size.

🚀 Usage

Clone the repository.

Ensure the initial puzzle configuration is set in the input.txt file (format R C followed by the initial matrix).

Run main.py and select the desired search algorithm via the console menu.